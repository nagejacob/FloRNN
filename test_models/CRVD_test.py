import sys
sys.path.append('..')
import argparse
from datasets import CRVDTestDataset
import numpy as np
import os
from models import ISP, FloRNNRaw
from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import structural_similarity
import torch
import torch.nn as nn
from utils.io import np2image_bgr

def raw_ssim(pack1, pack2):
    test_raw_ssim = 0
    for i in range(4):
        test_raw_ssim += structural_similarity(pack1[i], pack2[i], data_range=1.0)
    return test_raw_ssim / 4

def denoise_seq(seqn, a, b, model):
    T, C, H, W = seqn.shape
    a = a.expand((1, T, 1, H, W)).cuda()
    b = b.expand((1, T, 1, H, W)).cuda()
    seqdn = model(seqn.unsqueeze(0), a, b)[0]
    seqdn = torch.clamp(seqdn, 0, 1)
    return seqdn

def main(**args):
    dataset_val = CRVDTestDataset(CRVD_path=args['crvd_dir'])
    isp = ISP().cuda()
    isp.load_state_dict(torch.load(args['isp_path'])['state_dict'])

    if args['model'] == 'FloRNNRaw':
        model = FloRNNRaw(img_channels=4, num_resblocks=args['num_resblocks'], forward_count=args['forward_count'], border_ratio=args['border_ratio'])

    state_temp_dict = torch.load(args['model_file'])['state_dict']
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(state_temp_dict)
    model.eval()

    iso_psnr, iso_ssim = {}, {}
    for data in dataset_val:

        # our channels: RGGB, RViDeNet channels: RGBG. we must pass RGBG pack to ISP as it's pretrained by RViDeNet
        seq = data['seq'].cuda()
        seqn = data['seqn'].cuda()

        with torch.no_grad():
            seqdn = denoise_seq(seqn, data['a'], data['b'], model)
            seqn[:, 2:] = torch.flip(seqn[:, 2:], dims=[1])
            seqdn[:, 2:] = torch.flip(seqdn[:, 2:], dims=[1])
            seq[:, 2:] = torch.flip(seq[:, 2:], dims=[1])

        seq_raw_psnr, seq_srgb_psnr, seq_raw_ssim, seq_srgb_ssim = 0, 0, 0, 0
        for i in range(seq.shape[0]):
            gt_raw_frame = seq[i].cpu().numpy()
            denoised_raw_frame = (np.uint16(seqdn[i].cpu().numpy() * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (2 ** 12 - 1 - 240)
            with torch.no_grad():
                gt_srgb_frame = np.uint8(np.clip(isp(seq[i:i+1]).cpu().numpy()[0], 0, 1) * 255).astype(np.float32) / 255.
                denoised_srgb_frame = np.uint8(np.clip(isp(seqdn[i:i+1]).cpu().numpy()[0], 0, 1) * 255).astype(np.float32) / 255.

            seq_raw_psnr += compare_psnr(gt_raw_frame, denoised_raw_frame, data_range=1.0)
            seq_srgb_psnr += compare_psnr(gt_srgb_frame, denoised_srgb_frame, data_range=1.0)
            seq_raw_ssim += raw_ssim(gt_raw_frame, denoised_raw_frame)
            seq_srgb_ssim += structural_similarity(np.transpose(gt_srgb_frame, (1, 2, 0)), np.transpose(denoised_srgb_frame, (1, 2, 0)),
                                                   data_range=1.0, multichannel=True)

        seq_raw_psnr /= seq.shape[0]
        seq_srgb_psnr /= seq.shape[0]
        seq_raw_ssim /= seq.shape[0]
        seq_srgb_ssim /= seq.shape[0]

        if (str(data['iso'])+'raw') not in iso_psnr.keys():
            iso_psnr[str(data['iso'])+'raw'] = seq_raw_psnr / 5
            iso_psnr[str(data['iso'])+'srgb'] = seq_srgb_psnr / 5
            iso_ssim[str(data['iso'])+'raw'] = seq_raw_ssim / 5
            iso_ssim[str(data['iso'])+'srgb'] = seq_srgb_ssim / 5
        else:
            iso_psnr[str(data['iso'])+'raw'] += seq_raw_psnr / 5
            iso_psnr[str(data['iso']) + 'srgb'] += seq_srgb_psnr / 5
            iso_ssim[str(data['iso']) + 'raw'] += seq_raw_ssim / 5
            iso_ssim[str(data['iso']) + 'srgb'] += seq_srgb_ssim / 5

    dataset_raw_psnr, dataset_srgb_psnr, dataset_raw_ssim, dataset_srgb_ssim = 0, 0, 0, 0
    for iso in [1600, 3200, 6400, 12800, 25600]:
        print('iso %d, raw: %6.4f/%6.4f, srgb: %6.4f/%6.4f' % (iso, iso_psnr[str(iso)+'raw'], iso_ssim[str(iso)+'raw'],
                                                               iso_psnr[str(iso)+'srgb'], iso_ssim[str(iso)+'srgb']))
        dataset_raw_psnr += iso_psnr[str(iso)+'raw']
        dataset_srgb_psnr += iso_psnr[str(iso)+'srgb']
        dataset_raw_ssim += iso_ssim[str(iso)+'raw']
        dataset_srgb_ssim += iso_ssim[str(iso)+'srgb']

    print('CRVD, raw: %6.4f/%6.4f, srgb: %6.4f/%6.4f' % (dataset_raw_psnr / 5, dataset_raw_ssim / 5, dataset_srgb_psnr / 5, dataset_srgb_ssim / 5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test raw model")
    parser.add_argument("--model", type=str, default='FloRNNRaw') # model in ['FloRNNRaw']
    parser.add_argument("--num_resblocks", type=int, default=15)
    parser.add_argument("--forward_count", type=int, default=3)
    parser.add_argument("--border_ratio", type=float, default=0.1)
    parser.add_argument("--model_file", type=str, default='/home/nagejacob/Documents/codes/VDN/logs/ours_raw/ckpt_e12.pth')
    parser.add_argument("--crvd_dir", type=str, default="/hdd/Documents/datasets/CRVD")
    parser.add_argument("--isp_path", type=str, default="../models/rvidenet/isp.pth")
    argspar = parser.parse_args()

    print("\n### Testing model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))