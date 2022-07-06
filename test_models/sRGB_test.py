import sys
sys.path.append('..')
import argparse
from datasets import SrgbValDataset
from models import ForwardRNN, BiRNN, FloRNN, BasicVSRPlusPlus
import time
import torch
import torch.nn as nn
from utils.fastdvdnet_utils import fastdvdnet_batch_psnr
from utils.ssim import batch_ssim

def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    print(params / 1000 / 1000)
    return params

def denoise_seq(seqn, noise_std, model):

    # init arrays to handle contiguous frames and related patches
    numframes, C, H, W = seqn.shape

    # build noise map from noise std---assuming Gaussian noise
    noise_level_map = noise_std.expand((numframes, C, H, W)).cuda()

    with torch.no_grad():
        denframes = model(seqn.unsqueeze(0), noise_level_map.unsqueeze(0))

    denframes = torch.clamp(denframes.squeeze(0), 0., 1.)

    # free memory up
    del noise_level_map
    torch.cuda.empty_cache()

    # convert to appropiate type and return
    return denframes

def test(**args):
    test_set = SrgbValDataset(args['test_path'], num_input_frames=args['max_num_fr_per_seq'])

    if args['model'] == 'ForwardRNN':
        model = ForwardRNN(img_channels=3, num_resblocks=args['num_resblocks'])
    elif args['model'] == 'BiRNN':
        model = BiRNN(img_channels=3, num_resblocks=args['num_resblocks'])
    elif args['model'] == 'FloRNN':
        model = FloRNN(img_channels=3, num_resblocks=args['num_resblocks'], forward_count=args['forward_count'], border_ratio=args['border_ratio'])
    elif args['model'] == 'BasicVSRPlusPlus':
        model = BasicVSRPlusPlus(img_channels=3, spatial_blocks=6, temporal_blocks=6, num_channels=64)

    state_temp_dict = torch.load(args['model_file'])['state_dict']
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(state_temp_dict)
    model = model.module
    model.eval()

    dataset_psnr, dataset_ssim, seq_count = 0, 0, 0
    total_time, total_frames = 0, 0
    for data in test_set:
        seq = data['seq']

        # Add noise
        torch.manual_seed(0)
        noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma'])
        seqn = seq + noise
        noise_std = torch.FloatTensor([args['noise_sigma']])
        seqn = seqn.contiguous()

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            denframes = denoise_seq(seqn, noise_std=noise_std, model=model)

        torch.cuda.synchronize()
        total_time += time.time() - start_time
        total_frames += seqn.shape[0]

        psnr = fastdvdnet_batch_psnr(denframes, seq, 1.)
        ssim = batch_ssim(denframes, seq, 1.)
        dataset_psnr += psnr
        dataset_ssim += ssim
        seq_count += 1
        name = data['name'].split('/')[-2] + '/' + data['name'].split('/')[-1]
        print('{0:50}:, PSNR: {1:.4f}dB, SSIM: {2:.4f}'.format(name, psnr, ssim))
        if args['display_time']:
            print('frames: %d, time/frame: %6.4f s' % (total_frames, total_time / total_frames))

    print('sigma %d, PSNR: %6.4f, SSIM: %6.4f' % (int(round(args['noise_sigma'] * 255.)),
                                                  dataset_psnr/seq_count, dataset_ssim/seq_count))

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="test sRGB model")
    parser.add_argument("--model", type=str, default='BasicVSRPlusPlus') # model in ['ForwardRNN', 'BiRNN', 'FloRNN', 'BasciVSRPlusPlus']
    parser.add_argument("--num_resblocks", type=int, default=15)
    parser.add_argument("--forward_count", type=int, default=3)
    parser.add_argument("--border_ratio", type=float, default=0.1)
    parser.add_argument("--model_file", type=str, default='/home/nagejacob/Documents/codes/VDN/logs/basicvsr_plusplus/ckpt_e12.pth')
    parser.add_argument("--test_path", type=str, default="/hdd/Documents/datasets/Set8")
    parser.add_argument("--max_num_fr_per_seq", type=int, default=85)
    parser.add_argument("--noise_sigma", type=float, default=20, help='noise level used on test_models set')
    parser.add_argument("--display_time", type=bool, default=False)
    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.


    print("\n### Testing model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    for sigma in [10, 20, 30, 40, 50]:
        argspar.noise_sigma = sigma / 255.
        print('sigma=%d' % sigma)
        dataset_psnr = test(**vars(argspar))
