import sys
sys.path.append('..')
import argparse
from datasets import CRVDTrainDataset, CRVDTestDataset
from models import FloRNNRaw
import numpy as np
import os
from skimage.measure.simple_metrics import compare_psnr
import time
import torch
from torch.utils.data import DataLoader
from train_models.base_functions import batch_psnr, resume_training, save_model
from utils.io import log

torch.backends.cudnn.benchmark = True

def main(**args):
    dataset_train = CRVDTrainDataset(CRVD_path=args['CRVD_dir'],
                              patch_size=args['patch_size'],
                              patches_per_epoch=args['patches_per_epoch'],
                              mirror_seq=args['mirror_seq'])
    loader_train = DataLoader(dataset=dataset_train, batch_size=args['batch_size'], num_workers=4, shuffle=True, drop_last=True)
    dataset_val = CRVDTestDataset(CRVD_path=args['CRVD_dir'])

    if args['model'] == 'FloRNNRaw':
        model = FloRNNRaw(img_channels=4, num_resblocks=args['num_resblocks'], forward_count=args['forward_count'],
                          border_ratio=args['border_ratio'])
    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(model.module.trainable_parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=0.1)

    start_epoch = resume_training(args, model, optimizer, scheduler)
    for epoch in range(start_epoch, args['epochs']):
        start_time = time.time()

        # training
        model.train()
        for i, data in enumerate(loader_train):
            seq = data['seq'].cuda()
            N, T, C, H, W = seq.shape

            seqn = data['seqn'].cuda()
            a = data['a'].expand((N, T, 1, H, W)).cuda()
            b = data['b'].expand((N, T, 1, H, W)).cuda()

            seqdn = model(seqn, a, b)

            if args['model'] in ['FloRNNRaw']:
                end_index = -1 if (args['forward_count'] == -1) else (-args['forward_count'])
                loss = criterion(seq[:, 1:end_index], seqdn[:, 1:end_index]) / (N * 2)
            else:
                loss = criterion(seq, seqdn) / (N * 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % args['print_every'] == 0:
                train_psnr = torch.mean(batch_psnr(seq, seqdn)).item()
                log(args["log_file"], "[epoch {}][{}/{}] loss: {:1.4f} PSNR_train: {:1.4f}\n". \
                    format(epoch + 1, i + 1, int(args['patches_per_epoch'] // args['batch_size']), loss.item(), train_psnr))

        scheduler.step()

        # evaluating
        model.eval()
        iso_psnr = {}
        for data in dataset_val:
            seq = data['seq']
            T, C, H, W = seq.shape

            seqn = data['seqn'].cuda()
            a = data['a'].expand((T, 1, H, W)).cuda()
            b = data['b'].expand((T, 1, H, W)).cuda()

            with torch.no_grad():
                seqdn = torch.clamp(model(seqn.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0)).squeeze(0), 0., 1.)

            # calculate psnr the same as RViDeNet
            seq_psnr = 0
            for i in range(T):
                seq_psnr += compare_psnr(seq[i].numpy(),
                                         (np.uint16(seqdn[i].cpu().numpy() * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (2 ** 12 - 1 - 240),
                                         data_range=1.0)
            seq_psnr /= T

            if str(data['iso']) not in iso_psnr.keys():
                iso_psnr[str(data['iso'])] = seq_psnr
            else:
                iso_psnr[str(data['iso'])] += seq_psnr
        dataset_psnr = 0
        for iso in [1600, 3200, 6400, 12800, 25600]:
            log(args['log_file'], 'iso %d, %6.4f\n' % (iso, iso_psnr[str(iso)] / 5))
            dataset_psnr += iso_psnr[str(iso)] / 5
        dataset_psnr = dataset_psnr / 5

        log(args["log_file"], "\n[epoch %d] PSNR_val: %.4f, %0.2f hour/epoch\n\n" % (epoch + 1, dataset_psnr, (time.time()-start_time)/3600))

        # save model
        save_model(args, model, optimizer, scheduler, epoch + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")

    # Model parameters
    parser.add_argument("--model", type=str, default='FloRNNRaw')
    parser.add_argument("--num_resblocks", type=int, default=15)
    parser.add_argument("--forward_count", type=int, default=3)
    parser.add_argument("--border_ratio", type=float, default=0.1)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", "--e", type=int, default=12)
    parser.add_argument("--milestones", nargs=1, type=int, default=[11])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
    parser.add_argument("--patches_per_epoch", "--n", type=int, default=256000, help="Number of patches")
    parser.add_argument("--mirror_seq", type=bool, default=True)

    # Paths
    parser.add_argument("--CRVD_dir", type=str, default='/hdd/Documents/datasets/CRVD')
    parser.add_argument("--log_dir", type=str, default="../logs/FloRNNRaw")
    argspar = parser.parse_args()

    argspar.log_file = os.path.join(argspar.log_dir, 'log.out')

    if not os.path.exists(argspar.log_dir):
        os.makedirs(argspar.log_dir)
    log(argspar.log_file, "\n### Training the denoiser ###\n")
    log(argspar.log_file, "> Parameters:\n")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        log(argspar.log_file, '\t{}: {}\n'.format(p, v))

    main(**vars(argspar))