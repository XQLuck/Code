import os
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import torchvision
import random

from losses.multiscaleloss import *
from time import time

from core.test import test
from models.VGG19 import VGG19


def train(cfg, init_epoch, dataset_loader, train_transforms, val_transforms,deblurnet, deblurnet_solver, deblurnet_lr_scheduler,ckpt_dir, train_writer, val_writer,Best_Img_PSNR, Best_Epoch):


    n_itr = 0
    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        if epoch_idx >= 300  and epoch_idx <= 450:
            print("occlusion1")
            # deblurnet.confidencemap = deblurnet.module.confidencemap
            confidence_map_params_id = list(map(id, deblurnet.confidencemap.parameters()))
            base_params = filter(lambda x: id(x) not in confidence_map_params_id and x.requires_grad,deblurnet.parameters())
            new_params = filter(lambda x: id(x) in confidence_map_params_id and x.requires_grad, deblurnet.parameters())
            deblurnet_solver = torch.optim.Adam([{'params': base_params, 'lr': 1e-6},{'params': new_params, 'lr': 1e-4}],betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))       
        if epoch_idx > 450 and epoch_idx < 600:
            print("occlusion2")
            # deblurnet.confidencemap = deblurnet.module.confidencemap
            confidence_map_params_id = list(map(id, deblurnet.confidencemap.parameters()))
            base_params = filter(lambda x: id(x) not in confidence_map_params_id and x.requires_grad,deblurnet.parameters())
            new_params = filter(lambda x: id(x) in confidence_map_params_id and x.requires_grad, deblurnet.parameters())
            deblurnet_solver = torch.optim.Adam([{'params': base_params, 'lr': 1e-6},{'params': new_params, 'lr': 1e-5}],betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))
        if epoch_idx >= 600:
            print("occlusion3")
            # deblurnet.confidencemap = deblurnet.module.confidencemap
            confidence_map_params_id = list(map(id, deblurnet.confidencemap.parameters()))
            base_params = filter(lambda x: id(x) not in confidence_map_params_id and x.requires_grad,deblurnet.parameters())
            new_params = filter(lambda x: id(x) in confidence_map_params_id and x.requires_grad, deblurnet.parameters())
            deblurnet_solver = torch.optim.Adam([{'params': base_params, 'lr': 1e-6},{'params': new_params, 'lr': 1e-6}],betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))
        # Set up data loader
        train_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, train_transforms),
            batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)

        # Tick / tock
        epoch_start_time = time()
        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        deblur_mse_losses = utils.network_utils.AverageMeter()
        if cfg.TRAIN.USE_PERCET_LOSS == True:
            deblur_percept_losses = utils.network_utils.AverageMeter()
        deblur_losses = utils.network_utils.AverageMeter()
        img_PSNRs = utils.network_utils.AverageMeter()

        # Adjust learning rate
        deblurnet_lr_scheduler.step()
        print('[INFO] learning rate: {0}\n'.format(deblurnet_lr_scheduler.get_lr()))

        batch_end_time = time()
        seq_num = len(train_data_loader)

        vggnet = VGG19()
        if torch.cuda.is_available():
            vggnet = torch.nn.DataParallel(vggnet).cuda()

        for seq_idx, (_, seq_blur, seq_clear) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)
            # Get data from data loader
            seq_blur  = [utils.network_utils.var_or_cuda(img) for img in seq_blur]
            seq_clear = [utils.network_utils.var_or_cuda(img) for img in seq_clear]

            # switch models to training mode
            deblurnet.train()

            # Train the model
            last_img_blur = seq_blur[0]
            output_last_img = seq_blur[0]
            output_last_fea = None
            output_last_fea_down=None
            for batch_idx, [img_blur, img_clear] in enumerate(zip(seq_blur, seq_clear)):
                img_blur_hold = img_blur
                output_img, output_fea,output_last_fea_down_1 = deblurnet(img_blur, last_img_blur, output_last_img, output_last_fea,output_last_fea_down)


                deblur_mse_loss = mseLoss(output_img, img_clear)
                deblur_mse_losses.update(deblur_mse_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
                if cfg.TRAIN.USE_PERCET_LOSS == True:
                    deblur_percept_loss = perceptualLoss(output_img, img_clear, vggnet)
                    deblur_percept_losses.update(deblur_percept_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
                    deblur_loss = deblur_mse_loss + 0.01 * deblur_percept_loss
                else:
                    deblur_loss = deblur_mse_loss
                deblur_losses.update(deblur_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
                img_PSNR = PSNR(output_img, img_clear)
                img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)

                # deblurnet update
                deblurnet_solver.zero_grad()
                deblur_loss.backward()
                deblurnet_solver.step()

                # Append loss to TensorBoard
                train_writer.add_scalar('STFANet/DeblurLoss_0_TRAIN', deblur_loss.item(), n_itr)
                train_writer.add_scalar('STFANet/DeblurMSELoss_0_TRAIN', deblur_mse_loss.item(), n_itr)
                if cfg.TRAIN.USE_PERCET_LOSS == True:
                    train_writer.add_scalar('STFANet/DeblurPerceptLoss_0_TRAIN', deblur_percept_loss.item(), n_itr)
                n_itr = n_itr + 1

                # Tick / tock
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()

                last_img_blur = img_blur_hold
                output_last_img = output_img.clamp(0.0, 1.0).detach()
                output_last_fea = output_fea.detach()
                output_last_fea_down=output_last_fea_down_1.detach()


        train_writer.add_scalar('STFANet/EpochPSNR_0_TRAIN', img_PSNRs.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t ImgPSNR_avg {3}\n'.format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, img_PSNRs.avg))

        # Validate the training models
        if (epoch_idx + 1) % 2== 0:
            img_PSNR = test(cfg, epoch_idx, dataset_loader, val_transforms, deblurnet, val_writer)
            if img_PSNR >= Best_Img_PSNR:
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                Best_Img_PSNR = img_PSNR
                Best_Epoch = epoch_idx + 1
                utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), epoch_idx + 1, deblurnet, deblurnet_solver, Best_Img_PSNR, Best_Epoch)


        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                                                      epoch_idx + 1, deblurnet, deblurnet_solver, \
                                                      Best_Img_PSNR, Best_Epoch)



    train_writer.close()
    val_writer.close()