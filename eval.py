import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim



def _eval(model, args):
    state_dict = torch.load(args.test_model, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    adder = Adder()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():

        for iter_idx, data in enumerate(dataloader):
            input_img, input_img_d, input_img_g, label_img, name = data

            input_img = input_img.to(device)
            input_img_d = input_img_d.to(device)
            input_img_g = input_img_g.to(device)

            tm = time.time()

            pred = model(input_img, input_img_d, input_img_g)

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            label_img = (label_img).to(device)
            psnr_val = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            ssim_val = ssim(pred_clip, label_img, data_range=1, size_average=False)

            psnr_adder(psnr_val)
            ssim_adder(ssim_val)

            print('%d iter PSNR: %.4f ssim: %.4f time: %f' % (iter_idx + 1, psnr_val, ssim_val, elapsed))

            if args.save_image:
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                save_name = os.path.join(args.save_path, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

        print('==========================================================')
        print('The average PSNR is %.4f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f dB' % (ssim_adder.average()))
        print("Average time: %f" % adder.average())

