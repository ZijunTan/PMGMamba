import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f
from pytorch_msssim import ssim
from tqdm import tqdm


def _valid(model, args, ep):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    its = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():

        for idx, data in enumerate(tqdm(its)):
            input_img, input_img_d, input_img_g, label_img, _ = data
            input_img = input_img.to(device)
            input_img_d = input_img_d.to(device)
            input_img_g = input_img_g.to(device)
            label_img = label_img.to(device)

            # pred = model(input_img)[3]
            pred = model(input_img, input_img_d, input_img_g)

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            ssim_ = ssim(pred_clip, label_img, data_range=1, size_average=False)

            psnr_adder(psnr)
            ssim_adder(ssim_)

    model.train()
    return psnr_adder.average(), ssim_adder.average()
