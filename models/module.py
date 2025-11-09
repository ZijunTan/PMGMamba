import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x



class GSSM(nn.Module):
    def __init__(self, dim, d_state, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(nn.Conv2d(self.dim, hidden, 1, 1, 0))

        self.CPE = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden))

        self.route = nn.Sequential(nn.Linear(self.dim, self.dim // 3),
                                   nn.GELU(),
                                   nn.Linear(self.dim // 3, self.num_tokens),
                                   nn.LogSoftmax(dim=-1))

    def forward(self, x, x_size, global_weight):
        B, n, C = x.shape
        H, W = x_size

        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]
        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        #### SAR ####
        semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
        y = self.selectiveScan(semantic_x, global_weight)
        y = self.out_proj(self.out_norm(y))

        #### Invert ####
        x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, global_weight):

        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        ######## Global Enhancement ########
        Cs = Cs.float().view(B, K, -1, L) + global_weight
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, global_weight,  **kwargs):
        y = self.forward_core(x, global_weight)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class FeedForward(nn.Module):
    def __init__(self, dim, expand=2., bias=True):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * expand)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                 groups=hidden_features, bias=bias)

        self.dwconv3 = nn.Conv2d(hidden_features, 2, kernel_size=3, padding=1, bias=bias)

        self.dwconv4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x_in, x_p):

        x = self.project_in(x_in)
        x = self.dwconv(x)
        x_p = self.dwconv2(x_p)
        T, A = self.dwconv3(x_p).chunk(2, dim=1)
        T = self.sigmoid(T)
        x = x * T + A * (1 - T)
        x = F.gelu(self.dwconv4(x))
        x = self.project_out(x)

        return x



class PMGMamba_Block(nn.Module):
    def __init__(self, dim, d_state, inner_rank, num_tokens, mlp_ratio, norm_layer=nn.LayerNorm):
        super(PMGMamba_Block, self).__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.norm3 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.gssm = GSSM(self.dim, d_state, num_tokens=num_tokens, inner_rank=inner_rank, mlp_ratio=mlp_ratio)

        sample_rate = 2
        self.sampler = nn.MaxPool2d(kernel_size=sample_rate, stride=sample_rate)
        self.kernel_size = sample_rate
        self.patch_size = sample_rate
        self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size=self.kernel_size, padding=(self.kernel_size // sample_rate - 1),
                                            stride=sample_rate, groups=dim, bias=False)

        self.global_proj = nn.Linear(dim, d_state)
        self.gate = nn.Linear(dim, d_state)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm4 = norm_layer(dim)
        self.ffn = FeedForward(dim)
        self.conv2d = nn.Conv2d(int(dim * mlp_ratio), int(dim * mlp_ratio), kernel_size=3, stride=1, padding=1,
                                groups=dim, bias=False)


    def forward(self, x, x_p):

        ####GlobalGate####
        b, c, h, w = x.shape
        x_ = self.pool(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        global_x = self.global_proj(x_.mean(dim=1, keepdim=True)).permute(0, 2, 1).contiguous()
        global_gate = torch.sigmoid(self.gate(x_)).permute(0, 2, 1).contiguous()
        global_weight = (global_gate * global_x).unsqueeze(1)

        ####SAGMamba####
        xs = self.sampler(x)
        b, c, hs, ws = xs.shape
        xs_size = (hs, ws)
        xs = xs.view(b, c, -1).permute(0, 2, 1).contiguous()
        x_aca = self.gssm(self.norm3(xs), xs_size, global_weight) + xs
        x_aca = self.LocalProp(x_aca.permute(0, 2, 1).view(b, c, hs, ws).contiguous())
        x_aca = x_aca.view(b, c, -1).permute(0, 2, 1).contiguous()
        x = x_aca + self.scale2 + x.view(b, c, -1).permute(0, 2, 1).contiguous()

        ####PMGFFN####
        x_p = self.conv2d(x_p)
        x = self.ffn(self.norm4(x).permute(0, 2, 1).view(b, c, h, w).contiguous(), x_p) + x.permute(0, 2, 1).view(b, c, h, w).contiguous()

        return x



