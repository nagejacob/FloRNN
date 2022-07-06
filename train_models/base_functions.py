import glob
import math
import os
import re
import torch
from utils.io import log

def resume_training(args, model, optimizer, scheduler):
    """ Resumes previous training or starts anew
    """
    model_files = glob.glob(os.path.join(args['log_dir'], '*.pth'))

    if len(model_files) == 0:
        start_epoch = 0
    else:
        log(args.log_file, "> Resuming previous training\n")
        epochs_exist = []
        for model_file in model_files:
            result = re.findall('ckpt_e(.*).pth', model_file)
            epochs_exist.append(int(result[0]))
        max_epoch = max(epochs_exist)
        max_epoch_model_file = os.path.join(args['log_dir'], 'ckpt_e%d.pth' % max_epoch)
        checkpoint = torch.load(max_epoch_model_file)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = max_epoch

    return start_epoch

def save_model(args, model, optimizer, scheduler, epoch):
    save_dict = {
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler': scheduler.state_dict()}

    torch.save(save_dict, os.path.join(args['log_dir'], 'ckpt_e{}.pth'.format(epoch)))

# the same as skimage.metrics.peak_signal_noise_ratio
def batch_psnr(a, b):
    a = torch.clamp(a, 0, 1)
    b = torch.clamp(b, 0, 1)
    x = torch.mean((a - b) ** 2, dim=[-3, -2, -1])
    return 20 * torch.log(1 / torch.sqrt(x)) / math.log(10)
