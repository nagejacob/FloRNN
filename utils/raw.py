import torch

# simply convert raw seq to rgb seq for computing optical flow
def demosaic(raw_seq):
    N, T, C, H, W = raw_seq.shape
    rgb_seq = torch.empty((N, T, 3, H, W), dtype=raw_seq.dtype, device=raw_seq.device)
    rgb_seq[:, :, 0] = raw_seq[:, :, 0]
    rgb_seq[:, :, 1] = (raw_seq[:, :, 1] + raw_seq[:, :, 2]) / 2
    rgb_seq[:, :, 2] = (raw_seq[:, :, 3])
    return rgb_seq