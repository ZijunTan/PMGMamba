import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()

        self.layers = nn.ModuleList([
            ResBlock(out_channel)
            for i in range(num_res - 1)
        ])

    def forward(self, x, x_p):
        for layers in self.layers:
            x = layers(x, x_p)
        return x


class DBlock(nn.Module):
    def __init__(self, channel, num_res):
        super(DBlock, self).__init__()

        self.layers = nn.ModuleList([
            ResBlock(channel)
            for i in range(num_res - 1)
        ])

    def forward(self, x, x_p):
        for layers in self.layers:
            x = layers(x, x_p)
        return x


class PMGMamba(nn.Module):
    def __init__(self):
        super(PMGMamba, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([EBlock(base_channel, 4),
                                      EBlock(base_channel * 2, 4),
                                      EBlock(base_channel * 4, 4)])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)])

        self.Decoder = nn.ModuleList([DBlock(base_channel * 4, 4),
                                      DBlock(base_channel * 2, 4),
                                      DBlock(base_channel, 4)])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 1, base_channel, kernel_size=1, relu=True, stride=1)
        ])

        self.Convs_piror = nn.ModuleList([
            BasicConv(5, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(5, base_channel * 2 * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(5, base_channel * 4 * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(5, base_channel * 8 * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(5, base_channel * 4 * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(5, base_channel * 2 * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(5, base_channel * 2, kernel_size=3, relu=True, stride=1),
        ])

        self.neck = EBlock(base_channel * 8, 4)

    def forward(self, x, x_d, x_g):

        x_p = torch.cat((x, x_d, x_g), dim=1)
        x_p_1 = F.avg_pool2d(x_p, kernel_size=2, stride=2)
        x_p_2 = F.avg_pool2d(x_p, kernel_size=4, stride=4)
        x_p_3 = F.avg_pool2d(x_p, kernel_size=8, stride=8)

        # 256
        x_ = self.feat_extract[0](x)
        x_p_fea_0 = self.Convs_piror[0](x_p)
        res1 = self.Encoder[0](x_, x_p_fea_0)
        # 128
        z = self.feat_extract[1](res1)
        x_p_fea_1 = self.Convs_piror[1](x_p_1)
        res2 = self.Encoder[1](z, x_p_fea_1)
        # 64
        z = self.feat_extract[2](res2)
        x_p_fea_2 = self.Convs_piror[2](x_p_2)
        res3 = self.Encoder[2](z, x_p_fea_2)

        # 32 Neck
        z = self.feat_extract[3](res3)
        x_p_fea_3 = self.Convs_piror[3](x_p_3)
        z = self.neck(z, x_p_fea_3)

        # 64
        z = self.feat_extract[4](z)
        z = z + res3
        z = self.Convs[0](z)
        x_p_fea_4 = self.Convs_piror[4](x_p_2)
        z = self.Decoder[0](z, x_p_fea_4)
        # 128
        z = self.feat_extract[5](z)
        z = z + res2
        z = self.Convs[1](z)
        x_p_fea_5 = self.Convs_piror[5](x_p_1)
        z = self.Decoder[1](z, x_p_fea_5)
        # 256
        z = self.feat_extract[6](z)
        z = z + res1
        z = self.Convs[2](z)
        x_p_fea_6 = self.Convs_piror[6](x_p)
        z = self.Decoder[2](z, x_p_fea_6)
        z = self.feat_extract[7](z)

        return z + x



if __name__ == '__main__':
    torch.cuda.set_device(0)
    model = PMGMamba().cuda()
    x = torch.randn((1, 3, 256, 256)).cuda()
    d = torch.randn((1, 1, 256, 256)).cuda()
    g = torch.randn((1, 1, 256, 256)).cuda()
    out = model(x, d, g)
    print(out.shape)



