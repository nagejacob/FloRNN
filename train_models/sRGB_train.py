import sys
sys.path.append('..')
import argparse
from datasets import SrgbTrainDataset, SrgbValDataset
from models import ForwardRNN, BiRNN, FloRNN
import os
import time
import torch
from torch.utils.data import DataLoader
from train_models.base_functions import resume_training, save_model
from utils.fastdvdnet_utils import fastdvdnet_batch_psnr, normalize_augment
from utils.io import log

torch.backends.cudnn.benchmark = True

def main(**args):
    dataset_train = SrgbTrainDataset(seq_dir=args['trainset_dir'],
                              train_length=args['train_length'],
                              patch_size=args['patch_size'],
                              patches_per_epoch=args['patches_per_epoch'],
                              image_postfix='jpg',
                              pin_memory=True)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args['batch_size'], num_workers=4, shuffle=True, drop_last=True)
    dataset_val = SrgbValDataset(valsetdir=args['valset_dir'])
    loader_val = DataLoader(dataset=dataset_val, batch_size=1)

    if args['model'] == 'ForwardRNN':
        model = ForwardRNN(img_channels=3, num_resblocks=args['num_resblocks'])
    elif args['model'] == 'BiRNN':
        model = BiRNN(img_channels=3, num_resblocks=args['num_resblocks'])
    elif args['model'] == 'FloRNN':
        model = FloRNN(img_channels=3, num_resblocks=args['num_resblocks'], forward_count=args['forward_count'],
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
            seq = data['data'].cuda()
            seq = normalize_augment(seq)

            N, T, C, H, W = seq.shape
            stdn = torch.empty((N, 1, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
            noise_level_map = stdn.expand_as(seq)

            noise = torch.normal(mean=torch.zeros_like(seq), std=noise_level_map)
            seqn = seq + noise
            seqdn = model(seqn, noise_level_map)

            if args['model'] in ['FloRNN']:
                end_index = -1 if (args['forward_count'] == -1) else (-args['forward_count'])
                loss = criterion(seq[:, 1:end_index], seqdn[:, 1:end_index]) / (N * 2)
            else:
                loss = criterion(seq, seqdn) / (N * 2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i+1) % args['print_every'] == 0:
                train_psnr = fastdvdnet_batch_psnr(seq, seqdn)
                log(args["log_file"], "[epoch {}][{}/{}] loss: {:1.4f} PSNR_train: {:1.4f}\n". \
                    format(epoch + 1, i + 1, int(args['patches_per_epoch'] // args['batch_size']), loss.item(), train_psnr))

        scheduler.step()

        # evaluating
        model.eval()
        psnr_val = 0
        for i, data in enumerate(loader_val):
            seq = data['seq'].cuda()

            torch.manual_seed(0)
            stdn = torch.FloatTensor([args['val_noiseL']])
            noise_level_map = stdn.expand_as(seq)
            noise = torch.empty_like(seq).normal_(mean=0, std=args['val_noiseL'])
            seqn = seq + noise

            with torch.no_grad():
                seqdn = model(seqn, noise_level_map)
            psnr_val += fastdvdnet_batch_psnr(seq, seqdn)

        psnr_val = psnr_val / len(dataset_val)
        log(args["log_file"], "\n[epoch %d] PSNR_val: %.4f, %0.2f hour/epoch\n\n" % (epoch + 1, psnr_val, (time.time()-start_time)/3600))

        # save model
        save_model(args, model, optimizer, scheduler, epoch + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")

    # Model parameters
    parser.add_argument("--model", type=str, default='FloRNN')
    parser.add_argument("--num_resblocks", type=int, default=15)
    parser.add_argument("--forward_count", type=int, default=3)
    parser.add_argument("--border_ratio", type=float, default=0.1)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", "--e", type=int, default=12)
    parser.add_argument("--milestones", nargs=1, type=int, default=[11])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[0, 55])
    parser.add_argument("--val_noiseL", type=float, default=20)
    parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
    parser.add_argument("--patches_per_epoch", "--n", type=int, default=128000, help="Number of patches")

    # Paths
    parser.add_argument("--trainset_dir", type=str, default='/hdd/Documents/datasets/DAVIS-2017-trainval-480p')
    parser.add_argument("--valset_dir", type=str, default='/hdd/Documents/datasets/Set8')
    parser.add_argument("--log_dir", type=str, default="../logs/FloRNN")
    argspar = parser.parse_args()

    argspar.log_file = os.path.join(argspar.log_dir, 'log.out')
    argspar.train_length = 10 if (argspar.forward_count == -1) else (8 + argspar.forward_count)

    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    if not os.path.exists(argspar.log_dir):
        os.makedirs(argspar.log_dir)
    log(argspar.log_file, "\n### Training the denoiser ###\n")
    log(argspar.log_file, "> Parameters:\n")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        log(argspar.log_file, '\t{}: {}\n'.format(p, v))

    main(**vars(argspar))