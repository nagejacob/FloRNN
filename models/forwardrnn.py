from models.components import ResBlocks, D
from pytorch_pwc.extract_flow import extract_flow_torch
from pytorch_pwc.pwc import PWCNet
import torch
import torch.nn as nn
from utils.warp import warp

class ForwardRNN(nn.Module):
    def __init__(self, img_channels=3, num_resblocks=6, num_channels=64):
        super(ForwardRNN, self).__init__()
        self.num_channels = num_channels
        self.pwcnet = PWCNet()
        self.forward_rnn = ResBlocks(input_channels=img_channels + img_channels + num_channels, num_resblocks=num_resblocks, num_channels=num_channels)
        self.d = D(in_channels=num_channels, mid_channels=num_channels, out_channels=img_channels)

    def trainable_parameters(self):
        return [{'params':self.forward_rnn.parameters()}, {'params':self.d.parameters()}]

    def forward(self, seqn, noise_level_map):
        N, T, C, H, W = seqn.shape
        seqdn = torch.empty_like(seqn)

        init_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        h = self.forward_rnn(torch.cat((seqn[:, 0], noise_level_map[:, 0], init_h), dim=1))
        seqdn[:, 0] = self.d(h)

        for i in range(1, T):
            flow = extract_flow_torch(self.pwcnet, seqn[:, i], seqn[:, i-1])
            aligned_h, _ = warp(h, flow)
            h = self.forward_rnn(torch.cat((seqn[:, i], noise_level_map[:, i], aligned_h), dim=1))
            seqdn[:, i] = self.d(h)

        return seqdn
