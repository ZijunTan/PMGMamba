import os
import torch
from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from loss.vgg import PerceptualLoss
from loss.ssim import *
from warmup_scheduler import GradualWarmupScheduler
import numpy as np

def _train(model, args, logging):

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info("Total network parameters (excluding idr): %.2fM" % (total_params / 1e6))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.L1Loss()
    # vggloss = PerceptualLoss(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1


    ############ lossadder ############
    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_ssim_adder = Adder()
    epoch_total_adder = Adder()
    iter_pixel_adder = Adder()
    iter_ssim_adder = Adder()
    iter_total_adder = Adder()

    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = 0
    best_epoch = 0
    best_ssim = 0

    for epoch_idx in range(epoch, args.num_epoch + 1):
        logging.info("\n==> Name %s, Epoch %i, previous PSNR = %.4f, SSIM = %.4f in epoch %i" % (args.model_name, epoch_idx, best_psnr, best_ssim, best_epoch))
        epoch_timer.tic()
        iter_timer.tic()
        traintar = tqdm(dataloader, ncols=150)
        total_loss = 0
        l1_loss = 0
        ssim_loss = 0

        for iter_idx, batch_data in enumerate(traintar):

            input_img, input_img_d, input_img_g, label_img = batch_data
            input_img = input_img.to(device)
            input_img_d = input_img_d.to(device)
            input_img_g = input_img_g.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img, input_img_d, input_img_g)
            l4 = criterion(pred_img, label_img)
            loss_content = l4

            ###########   SSimloss
            s4 = 1 - torch.mean(ssim(pred_img, label_img))
            loss_ssim = s4

            loss = loss_content + 0.1 * loss_ssim
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_ssim_adder(loss_ssim.item())
            iter_total_adder(loss.item())

            epoch_pixel_adder(loss_content.item())
            epoch_ssim_adder(loss_ssim.item())
            epoch_total_adder(loss.item())

            total_loss += loss.item()
            l1_loss += loss_content.item()
            ssim_loss += loss_ssim.item()

            traintar.set_description("Total_Loss: %.4f, L1_Loss: %.4f, SSim_Loss: %.4f, LR: %.6f" %
                                     (total_loss / (iter_idx + 1), l1_loss / (iter_idx + 1),
                                      ssim_loss / (iter_idx + 1), scheduler.get_lr()[0]))

            if (iter_idx + 1) % args.print_freq == 0:
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_ssim_adder.reset()
                iter_total_adder.reset()

        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        logging.info("Epoch: %i ==> Elapsed time: %.4f Total loss: %.4f L1 Loss: %.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_total_adder.average(), epoch_pixel_adder.average()))

        epoch_pixel_adder.reset()
        epoch_ssim_adder.reset()
        epoch_total_adder.reset()
        scheduler.step()

        val, ssim_ = _valid(model, args, epoch_idx)
        logging.info('PSNR: %.4f' % (val))
        writer.add_scalar('PSNR', val, epoch_idx)
        if val >= best_psnr:
            best_psnr = val
            best_ssim = ssim_
            best_epoch = epoch_idx
            torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))

