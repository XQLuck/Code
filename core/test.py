

import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multiscaleloss import *
import torchvision

import numpy as np
import scipy.io as io

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def test(cfg, epoch_idx, dataset_loader, test_transforms, deblurnet, test_writer):
    # Set up data loader
    test_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, test_transforms),
        batch_size=cfg.CONST.TEST_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)

    seq_num = len(test_data_loader)
    # Batch average meterics
    img_PSNRs = utils.network_utils.AverageMeter()
    test_psnr = dict()
    g_names= 'init'

    for seq_idx, (name, seq_blur, seq_clear) in enumerate(test_data_loader):
        

        seq_blur = [utils.network_utils.var_or_cuda(img) for img in seq_blur]
        seq_clear = [utils.network_utils.var_or_cuda(img) for img in seq_clear]
        seq_len = len(seq_blur)
        # Switch models to training mode
        deblurnet.eval()

        if cfg.NETWORK.PHASE == 'test':
            if not g_names == name[0]:
                g_names = name[0]
                save_num = 0

            assert (len(name) == 1)
            name = name[0]
            if not name in test_psnr:
                test_psnr[name] = {
                    'n_samples': 0,
                    'psnr': []
                }
        with torch.no_grad():
            last_img_blur = seq_blur[0]
            output_last_img = seq_blur[0]
            output_last_fea = None
            output_last_fea_down=None
            for batch_idx, [img_blur, img_clear] in enumerate(zip(seq_blur, seq_clear)):
                img_blur_hold = img_blur

                torch.cuda.synchronize()

                output_img, output_fea,output_last_fea_down_1= deblurnet(img_blur, last_img_blur, output_last_img, output_last_fea,output_last_fea_down)
                torch.cuda.synchronize()
                img_PSNR = PSNR(output_img, img_clear)
                img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)

                if seq_idx == 0 and batch_idx < cfg.TEST.VISUALIZATION_NUM and not cfg.NETWORK.PHASE == 'test':
                    if epoch_idx == 0 or cfg.NETWORK.PHASE in ['test','resume']:
                        img_blur = img_blur[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                        img_clear = img_clear[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                        test_writer.add_image('STFANet/IMG_BLUR' + str(batch_idx + 1), img_blur, epoch_idx + 1)
                        test_writer.add_image('STFANet/IMG_CLEAR' + str(batch_idx + 1), img_clear, epoch_idx + 1)

                        output_last_img = output_last_img[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    img_out = output_img[0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)

                    test_writer.add_image('STFANet/LAST_IMG_OUT' +str(batch_idx + 1), output_last_img, epoch_idx + 1)
                    test_writer.add_image('STFANet/IMAGE_OUT' +str(batch_idx + 1), img_out, epoch_idx + 1)

                if cfg.NETWORK.PHASE == 'test':
                    test_psnr[name]['n_samples'] += 1
                    test_psnr[name]['psnr'].append(img_PSNR)
                    img_dir = os.path.join(cfg.DIR.OUT_PATH, name)
                    if not os.path.isdir(img_dir):
                        mkdir(img_dir)

                last_img_blur = img_blur_hold
                output_last_img = output_img.clamp(0.0, 1.0)
                output_last_fea = output_fea
                output_last_fea_down=output_last_fea_down_1




    # Output testing results
    if cfg.NETWORK.PHASE == 'test':

        # Output test results
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
        for name in test_psnr:
            test_psnr[name]['psnr'] =  torch.mean(torch.stack(test_psnr[name]['psnr'],0))
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],test_psnr[name]['psnr']))

        result_file = open(os.path.join(cfg.DIR.OUT_PATH, 'test_result.txt'), 'w')
        sys.stdout = result_file
        print('============================ TEST RESULTS ============================')
       
        for name in test_psnr:
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))
        result_file.close()
    else:
        # Output val results
        print('============================ TEST RESULTS ============================')
        print('ImgPSNR_avg :',img_PSNRs.avg)

        # Add testing results to TensorBoard
        test_writer.add_scalar('STFANet/EpochPSNR_1_TEST', img_PSNRs.avg, epoch_idx + 1)

        return (img_PSNRs.avg)