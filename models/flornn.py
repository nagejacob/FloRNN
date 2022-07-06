from models.components import ResBlocks, D
from pytorch_pwc.extract_flow import extract_flow_torch
from pytorch_pwc.pwc import PWCNet
from softmax_splatting.softsplat import FunctionSoftsplat
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.warp import warp

def expand(ten, size_h, size_w, value=0):
    return F.pad(ten, pad=[size_w, size_w, size_h, size_h], mode='constant', value=value)

def split_border(ten, size_h, size_w):
    img = ten[:, :, size_h:-size_h, size_w:-size_w]
    return img, ten

def merge_border(img, border, size_h, size_w):
    expanded_img = F.pad(img, pad=[size_w, size_w, size_h, size_h], mode='constant')
    expanded_img[:, :, :size_h, :] = border[:, :, :size_h, :]
    expanded_img[:, :, -size_h:, :] = border[:, :, -size_h:, :]
    expanded_img[:, :, :, :size_w] = border[:, :, :, :size_w]
    expanded_img[:, :, :, -size_w:] = border[:, :, :, -size_w:]
    return expanded_img

class FloRNN(nn.Module):
    def __init__(self, img_channels, num_resblocks=6, num_channels=64, forward_count=2, border_ratio=0.3):
        super(FloRNN, self).__init__()
        self.num_channels = num_channels
        self.forward_count = forward_count
        self.pwcnet = PWCNet()
        self.forward_rnn = ResBlocks(input_channels=img_channels + img_channels + num_channels, num_resblocks=num_resblocks, num_channels=num_channels)
        self.backward_rnn = ResBlocks(input_channels=img_channels + img_channels + num_channels, num_resblocks=num_resblocks, num_channels=num_channels)
        self.d = D(in_channels=num_channels * 2, mid_channels=num_channels * 2, out_channels=img_channels)
        self.border_ratio = border_ratio

    def trainable_parameters(self):
        return [{'params':self.forward_rnn.parameters()}, {'params':self.backward_rnn.parameters()}, {'params':self.d.parameters()}]

    def forward(self, seqn_not_pad, noise_level_map_not_pad):
        N, T, C, H, W = seqn_not_pad.shape
        seqdn = torch.empty_like(seqn_not_pad)
        expanded_forward_flow_queue = []
        border_queue = []
        size_h, size_w = int(H * self.border_ratio), int(W * self.border_ratio)

        # reflect pad seqn and noise_level_map
        seqn = torch.empty((N, T+self.forward_count, C, H, W), device=seqn_not_pad.device)
        noise_level_map = torch.empty((N, T+self.forward_count, C, H, W), device=noise_level_map_not_pad.device)
        seqn[:, :T] = seqn_not_pad
        noise_level_map[:, :T] = noise_level_map_not_pad
        for i in range(self.forward_count):
            seqn[:, T+i] = seqn_not_pad[:, T-2-i]
            noise_level_map[:, T+i] = noise_level_map_not_pad[:, T-2-i]

        init_backward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        backward_h = self.backward_rnn(torch.cat((seqn[:, 0], noise_level_map[:, 0], init_backward_h), dim=1))
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        forward_h = self.forward_rnn(torch.cat((seqn[:, 0], noise_level_map[:, 0], init_forward_h), dim=1))

        for i in range(1, T+self.forward_count):
            forward_flow = extract_flow_torch(self.pwcnet, seqn[:, i-1], seqn[:, i])

            expanded_backward_h, expanded_forward_flow = expand(backward_h, size_h, size_w), expand(forward_flow, size_h, size_w)
            expanded_forward_flow_queue.append(expanded_forward_flow)
            aligned_expanded_backward_h = FunctionSoftsplat(expanded_backward_h, expanded_forward_flow, None, 'average')
            aligned_backward_h, border = split_border(aligned_expanded_backward_h, size_h, size_w)
            border_queue.append(border)

            backward_h = self.backward_rnn(torch.cat((seqn[:, i], noise_level_map[:, i], aligned_backward_h), dim=1))

            if i >= self.forward_count:
                aligned_backward_h = backward_h
                for j in reversed(range(self.forward_count)):
                    aligned_backward_h = merge_border(aligned_backward_h, border_queue[j], size_h, size_w)
                    aligned_backward_h, _ = warp(aligned_backward_h, expanded_forward_flow_queue[j])
                    aligned_backward_h, _ = split_border(aligned_backward_h, size_h, size_w)

                seqdn[:, i - self.forward_count] = self.d(torch.cat((forward_h, aligned_backward_h), dim=1))

                backward_flow = extract_flow_torch(self.pwcnet, seqn[:, i-self.forward_count+1], seqn[:, i-self.forward_count])
                aligned_forward_h, _ = warp(forward_h, backward_flow)
                forward_h = self.forward_rnn(torch.cat((seqn[:, i-self.forward_count+1], noise_level_map[:, i-self.forward_count+1], aligned_forward_h), dim=1))
                expanded_forward_flow_queue.pop(0)
                border_queue.pop(0)

        return seqdn

