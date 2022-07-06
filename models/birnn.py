from models.components import ResBlocks, D
from pytorch_pwc.extract_flow import extract_flow_torch
from pytorch_pwc.pwc import PWCNet
import torch
import torch.nn as nn
from utils.warp import warp

class BiRNN(nn.Module):
    def __init__(self, img_channels=3, num_resblocks=6, num_channels=64):
        super(BiRNN, self).__init__()
        self.num_channels = num_channels
        self.pwcnet = PWCNet()
        self.forward_rnn = ResBlocks(input_channels=img_channels + img_channels + num_channels, num_resblocks=num_resblocks, num_channels=num_channels)
        self.backward_rnn = ResBlocks(input_channels=img_channels + img_channels + num_channels, num_resblocks=num_resblocks, num_channels=num_channels)
        self.d = D(in_channels=num_channels * 2, mid_channels=num_channels * 2, out_channels=img_channels)

    def trainable_parameters(self):
        return [{'params':self.forward_rnn.parameters()}, {'params':self.backward_rnn.parameters()}, {'params':self.d.parameters()}]

    def forward(self, seqn, noise_level_map):
        if self.training:
            feature_device = torch.device('cuda')
        else:
            feature_device = torch.device('cpu')
        N, T, C, H, W = seqn.shape
        forward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        backward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        seqdn = torch.empty_like(seqn)

        # extract forward features
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        forward_h = self.forward_rnn(torch.cat((seqn[:, 0], noise_level_map[:, 0], init_forward_h), dim=1))
        forward_hs[:, 0] = forward_h.to(feature_device)
        for i in range(1, T):
            flow = extract_flow_torch(self.pwcnet, seqn[:, i], seqn[:, i-1])
            aligned_forward_h, _ = warp(forward_h, flow)
            forward_h = self.forward_rnn(torch.cat((seqn[:, i], noise_level_map[:, i], aligned_forward_h), dim=1))
            forward_hs[:, i] = forward_h.to(feature_device)

        # extract backward features
        init_backward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        backward_h = self.backward_rnn(torch.cat((seqn[:, -1], noise_level_map[:, -1], init_backward_h), dim=1))
        backward_hs[:, -1] = backward_h.to(feature_device)
        for i in range(2, T+1):
            flow = extract_flow_torch(self.pwcnet, seqn[:, T-i], seqn[:, T-i+1])
            aligned_backward_h, _ = warp(backward_h, flow)
            backward_h = self.backward_rnn(torch.cat((seqn[:, T-i], noise_level_map[:, T-i], aligned_backward_h), dim=1))
            backward_hs[:, T-i] = backward_h.to(feature_device)

        # generate results
        for i in range(T):
            seqdn[:, i] = self.d(torch.cat((forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device)), dim=1))

        return seqdn
