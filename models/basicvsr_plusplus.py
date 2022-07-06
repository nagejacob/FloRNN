from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from models.components import ResBlocks, D
from pytorch_pwc.extract_flow import extract_flow_torch
from pytorch_pwc.pwc import PWCNet
import torch
import torch.nn as nn
from utils.warp import warp

class BasicVSRPlusPlus(nn.Module):
    def __init__(self, img_channels=3, spatial_blocks=-1, temporal_blocks=-1, num_channels=64):
        super(BasicVSRPlusPlus, self).__init__()
        self.num_channels = num_channels
        self.pwcnet = PWCNet()

        self.feat_extract = ResBlocks(input_channels=img_channels * 2, num_resblocks=spatial_blocks, num_channels=num_channels)

        self.backbone = nn.ModuleDict()
        self.deform_align = nn.ModuleDict()
        self.module_names = ['forward_1', 'backward_1', 'forward_2', 'backward_2']
        for i, module_name in enumerate(self.module_names):
            self.backbone[module_name] = ResBlocks(input_channels=num_channels * (i+2), num_resblocks=temporal_blocks, num_channels=num_channels)
            self.deform_align[module_name] = SecondOrderDeformableAlignment(
                2 * num_channels,
                num_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=10)

        self.d = D(in_channels=num_channels * 4, mid_channels=num_channels * 2, out_channels=img_channels)
        self.device = torch.device('cuda')

    def trainable_parameters(self):
        return [{'params':self.feat_extract.parameters()}, {'params':self.backbone.parameters()},
                {'params':self.deform_align.parameters()}, {'params':self.d.parameters()}]

    def spatial_feature(self, seqn, noise_level_map):
        spatial_hs = []
        for i in range(seqn.shape[1]):
            spatial_h = self.feat_extract(torch.cat((seqn[:, i].cuda(), noise_level_map[:, i].cuda()), dim=1))
            if not self.training:
                spatial_h = spatial_h.cpu()
            spatial_hs.append(spatial_h)
        return spatial_hs

    def extract_flows(self, seqn):
        N, T, C, H, W = seqn.shape
        forward_flows, backward_flows = [], []
        for i in range(T-1):
            forward_flow = extract_flow_torch(self.pwcnet, seqn[:, i+1].cuda(), seqn[:, i].cuda())
            backward_flow = extract_flow_torch(self.pwcnet, seqn[:, i].cuda(), seqn[:, i+1].cuda())
            if not self.training:
                forward_flow = forward_flow.cpu()
                backward_flow = backward_flow.cpu()
            forward_flows.append(forward_flow)
            backward_flows.append(backward_flow)
        return forward_flows, backward_flows

    def forward(self, seqn, noise_level_map):
        if self.training:
            self.device = torch.device('cuda')
            return self.forward_train(seqn, noise_level_map)
        else:
            self.device = torch.device('cpu')
            return self.forward_test(seqn, noise_level_map)

    def forward_train(self, seqn, noise_level_map):
        N, T, C, H, W = seqn.shape
        hs = {}
        for module_name in self.module_names:
            hs[module_name] = [None] * T
        zeros_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        zeros_flow = torch.zeros((N, 2, H, W), device=seqn.device)
        seqdn = torch.empty_like(seqn)

        # extract flows
        forward_flows, backward_flows = self.extract_flows(seqn)

        # extract spatial features
        hs['spatial'] = self.spatial_feature(seqn, noise_level_map)

        # extract forward features
        spatial_h = hs['spatial'][0]
        forward_h = self.backbone['forward_1'](torch.cat((spatial_h, zeros_h), dim=1))
        hs['forward_1'][0] = forward_h

        spatial_h = hs['spatial'][1]
        flow_n1 = forward_flows[0]
        forward_h_n1 = forward_h
        aligned_forward_h_n1, _ = warp(forward_h_n1, flow_n1)
        feat_prop = self.deform_align['forward_1'](torch.cat((forward_h_n1, zeros_h), dim=1),
                                                   torch.cat((aligned_forward_h_n1, spatial_h, zeros_h), dim=1),
                                                   flow_n1, zeros_flow)
        forward_h = self.backbone['forward_1'](torch.cat((spatial_h, feat_prop), dim=1))
        hs['forward_1'][1] = forward_h

        for i in range(2, T):
            spatial_h = hs['spatial'][i]
            flow_n1 = forward_flows[i - 1]
            forward_h_n1 = forward_h
            aligned_forward_h_n1, _ = warp(forward_h_n1, flow_n1)
            flow_n2 = flow_n1 + warp(forward_flows[i - 2], flow_n1)[0]
            forward_h_n2 = hs['forward_1'][i - 2]
            aligned_forward_h_n2, _ = warp(forward_h_n2, flow_n2)
            feat_prop = self.deform_align['forward_1'](torch.cat((forward_h_n1, forward_h_n2), dim=1), torch.cat(
                (aligned_forward_h_n1, spatial_h, aligned_forward_h_n2), dim=1), flow_n1, flow_n2)
            forward_h = self.backbone['forward_1'](torch.cat((spatial_h, feat_prop), dim=1))
            hs['forward_1'][i] = forward_h

        # extract backward features
        spatial_h = hs['spatial'][-1]
        backward_h = self.backbone['backward_1'](torch.cat((spatial_h, zeros_h, hs['forward_1'][-1]), dim=1))
        hs['backward_1'][-1] = backward_h

        spatial_h = hs['spatial'][-2]
        flow_p1 = backward_flows[-1]
        backward_h_p1 = backward_h
        aligned_backward_h_p1, _ = warp(backward_h_p1, flow_p1)
        feat_prop = self.deform_align['backward_1'](torch.cat((backward_h_p1, zeros_h), dim=1),
                                                    torch.cat((aligned_backward_h_p1, spatial_h, zeros_h), dim=1),
                                                    flow_p1, zeros_flow)
        backward_h = self.backbone['backward_1'](torch.cat((spatial_h, feat_prop, hs['forward_1'][-2]), dim=1))
        hs['backward_1'][-2] = backward_h

        for i in range(3, T + 1):
            spatial_h = hs['spatial'][T - i]
            flow_p1 = backward_flows[T - i]
            backward_h_p1 = backward_h
            aligned_backward_h_p1, _ = warp(backward_h_p1, flow_p1)
            flow_p2 = flow_p1 + warp(backward_flows[T - i + 1], flow_p1)[0]
            backward_h_p2 = hs['backward_1'][T - i + 1]
            aligned_backward_h_p2, _ = warp(backward_h_p2, flow_p2)
            feat_prop = self.deform_align['backward_1'](torch.cat((backward_h_p1, backward_h_p2), dim=1),
                                                        torch.cat((aligned_backward_h_p1, spatial_h, backward_h_p2),
                                                                  dim=1), flow_p1, flow_p2)
            backward_h = self.backbone['backward_1'](
                torch.cat((spatial_h, feat_prop, hs['forward_1'][T - i]), dim=1))
            hs['backward_1'][T - i] = backward_h

        # extract forward features
        spatial_h = hs['spatial'][0]
        forward_h = self.backbone['forward_2'](torch.cat((spatial_h, zeros_h,
                                                          hs['forward_1'][0],
                                                          hs['backward_1'][0]), dim=1))
        hs['forward_2'][0] = forward_h

        spatial_h = hs['spatial'][1]
        flow_n1 = forward_flows[0]
        forward_h_n1 = forward_h
        aligned_forward_h_n1, _ = warp(forward_h_n1, flow_n1)
        feat_prop = self.deform_align['forward_2'](torch.cat((forward_h_n1, zeros_h), dim=1),
                                                   torch.cat((aligned_forward_h_n1, spatial_h, zeros_h), dim=1),
                                                   flow_n1, zeros_flow)
        forward_h = self.backbone['forward_2'](
            torch.cat((spatial_h, feat_prop, hs['forward_1'][1], hs['backward_1'][1]), dim=1))
        hs['forward_2'][1] = forward_h

        for i in range(2, T):
            spatial_h = hs['spatial'][i]
            flow_n1 = forward_flows[i - 1]
            forward_h_n1 = forward_h
            aligned_forward_h_n1, _ = warp(forward_h_n1, flow_n1)
            flow_n2 = flow_n1 + warp(forward_flows[i - 2], flow_n1)[0]
            forward_h_n2 = hs['forward_2'][i - 2]
            aligned_forward_h_n2, _ = warp(forward_h_n2, flow_n2)
            feat_prop = self.deform_align['forward_2'](torch.cat((forward_h_n1, forward_h_n2), dim=1), torch.cat(
                (aligned_forward_h_n1, spatial_h, aligned_forward_h_n2), dim=1), flow_n1, flow_n2)
            forward_h = self.backbone['forward_2'](
                torch.cat((spatial_h, feat_prop, hs['forward_1'][i], hs['backward_1'][i]), dim=1))
            hs['forward_2'][i] = forward_h

        # extract backward features
        spatial_h = hs['spatial'][-1]
        backward_h = self.backbone['backward_2'](
            torch.cat((spatial_h, zeros_h, hs['forward_1'][-1], hs['backward_1'][-1], hs['forward_2'][-1]),
                      dim=1))
        hs['backward_2'][-1] = backward_h

        spatial_h = hs['spatial'][-2]
        flow_p1 = backward_flows[-1]
        backward_h_p1 = backward_h
        aligned_backward_h_p1, _ = warp(backward_h_p1, flow_p1)
        feat_prop = self.deform_align['backward_2'](torch.cat((backward_h_p1, zeros_h), dim=1),
                                                    torch.cat((aligned_backward_h_p1, spatial_h, zeros_h), dim=1),
                                                    flow_p1, zeros_flow)
        backward_h = self.backbone['backward_2'](
            torch.cat((spatial_h, feat_prop, hs['forward_1'][-2], hs['backward_1'][-2], hs['forward_2'][-2]),
                      dim=1))
        hs['backward_2'][-2] = backward_h

        for i in range(3, T + 1):
            spatial_h = hs['spatial'][T - i]
            flow_p1 = backward_flows[T - i]
            backward_h_p1 = backward_h
            aligned_backward_h_p1, _ = warp(backward_h_p1, flow_p1)
            flow_p2 = flow_p1 + warp(backward_flows[T - i + 1], flow_p1)[0]
            backward_h_p2 = hs['backward_2'][T - i + 1]
            aligned_backward_h_p2, _ = warp(backward_h_p2, flow_p2)
            feat_prop = self.deform_align['backward_2'](torch.cat((backward_h_p1, backward_h_p2), dim=1),
                                                        torch.cat((aligned_backward_h_p1, spatial_h, backward_h_p2),
                                                                  dim=1), flow_p1, flow_p2)
            backward_h = self.backbone['backward_2'](torch.cat((spatial_h, feat_prop, hs['forward_1'][T - i],
                                                                hs['backward_1'][T - i], hs['forward_2'][T - i]),
                                                               dim=1))
            hs['backward_2'][T - i] = backward_h

        # generate results
        for i in range(T):
            seqdn[:, i] = self.d(torch.cat((hs['forward_1'][i], hs['backward_1'][i], hs['forward_2'][i], hs['backward_2'][i]), dim=1))

        return seqdn

    def forward_test(self, seqn, noise_level_map):
        N, T, C, H, W = seqn.shape
        hs = {}
        for module_name in self.module_names:
            hs[module_name] = [None] * T
        zeros_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        zeros_flow = torch.zeros((1, 2, H, W), device=seqn.device)
        seqdn = torch.empty_like(seqn)

        # extract flows
        forward_flows, backward_flows = self.extract_flows(seqn)

        # extract spatial features
        hs['spatial'] = self.spatial_feature(seqn, noise_level_map)

        # extract forward features
        spatial_h = hs['spatial'][0].cuda()
        forward_h = self.backbone['forward_1'](torch.cat((spatial_h, zeros_h.cuda()), dim=1))
        hs['forward_1'][0] = forward_h.cpu()

        spatial_h = hs['spatial'][1].cuda()
        flow_n1 = forward_flows[0].cuda()
        forward_h_n1 = forward_h
        aligned_forward_h_n1, _ = warp(forward_h_n1, flow_n1)
        feat_prop = self.deform_align['forward_1'](torch.cat((forward_h_n1, zeros_h.cuda()), dim=1),
                                                   torch.cat((aligned_forward_h_n1, spatial_h, zeros_h.cuda()), dim=1),
                                                   flow_n1, zeros_flow.cuda())
        forward_h = self.backbone['forward_1'](torch.cat((spatial_h, feat_prop), dim=1))
        hs['forward_1'][1] = forward_h.cpu()

        for i in range(2, T):
            spatial_h = hs['spatial'][i].cuda()
            flow_n1 = forward_flows[i - 1].cuda()
            forward_h_n1 = forward_h
            aligned_forward_h_n1, _ = warp(forward_h_n1, flow_n1)
            flow_n2 = flow_n1 + warp(forward_flows[i - 2].cuda(), flow_n1)[0]
            forward_h_n2 = hs['forward_1'][i - 2].cuda()
            aligned_forward_h_n2, _ = warp(forward_h_n2, flow_n2)
            feat_prop = self.deform_align['forward_1'](torch.cat((forward_h_n1, forward_h_n2), dim=1), torch.cat(
                (aligned_forward_h_n1, spatial_h, aligned_forward_h_n2), dim=1), flow_n1, flow_n2)
            forward_h = self.backbone['forward_1'](torch.cat((spatial_h, feat_prop), dim=1))
            hs['forward_1'][i] = forward_h.cpu()

        # extract backward features
        spatial_h = hs['spatial'][-1].cuda()
        backward_h = self.backbone['backward_1'](torch.cat((spatial_h, zeros_h.cuda(), hs['forward_1'][-1].cuda()), dim=1))
        hs['backward_1'][-1] = backward_h.cpu()

        spatial_h = hs['spatial'][-2].cuda()
        flow_p1 = backward_flows[-1].cuda()
        backward_h_p1 = backward_h
        aligned_backward_h_p1, _ = warp(backward_h_p1, flow_p1)
        feat_prop = self.deform_align['backward_1'](torch.cat((backward_h_p1, zeros_h.cuda()), dim=1),
                                                    torch.cat((aligned_backward_h_p1, spatial_h, zeros_h.cuda()), dim=1),
                                                    flow_p1, zeros_flow.cuda())
        backward_h = self.backbone['backward_1'](torch.cat((spatial_h, feat_prop, hs['forward_1'][-2].cuda()), dim=1))
        hs['backward_1'][-2] = backward_h.cpu()

        for i in range(3, T + 1):
            spatial_h = hs['spatial'][T - i].cuda()
            flow_p1 = backward_flows[T - i].cuda()
            backward_h_p1 = backward_h
            aligned_backward_h_p1, _ = warp(backward_h_p1, flow_p1)
            flow_p2 = flow_p1 + warp(backward_flows[T - i + 1].cuda(), flow_p1)[0]
            backward_h_p2 = hs['backward_1'][T - i + 1].cuda()
            aligned_backward_h_p2, _ = warp(backward_h_p2, flow_p2)
            feat_prop = self.deform_align['backward_1'](torch.cat((backward_h_p1, backward_h_p2), dim=1),
                                                        torch.cat((aligned_backward_h_p1, spatial_h, backward_h_p2),
                                                                  dim=1), flow_p1, flow_p2)
            backward_h = self.backbone['backward_1'](
                torch.cat((spatial_h, feat_prop, hs['forward_1'][T - i].cuda()), dim=1))
            hs['backward_1'][T - i] = backward_h.cpu()

        # extract forward features
        spatial_h = hs['spatial'][0].cuda()
        forward_h = self.backbone['forward_2'](torch.cat((spatial_h, zeros_h.cuda(),
                                                          hs['forward_1'][0].cuda(),
                                                          hs['backward_1'][0].cuda()), dim=1))
        hs['forward_2'][0] = forward_h.cpu()

        spatial_h = hs['spatial'][1].cuda()
        flow_n1 = forward_flows[0].cuda()
        forward_h_n1 = forward_h
        aligned_forward_h_n1, _ = warp(forward_h_n1, flow_n1)
        feat_prop = self.deform_align['forward_2'](torch.cat((forward_h_n1, zeros_h.cuda()), dim=1),
                                                   torch.cat((aligned_forward_h_n1, spatial_h, zeros_h.cuda()), dim=1),
                                                   flow_n1, zeros_flow.cuda())
        forward_h = self.backbone['forward_2'](
            torch.cat((spatial_h, feat_prop, hs['forward_1'][1].cuda(), hs['backward_1'][1].cuda()), dim=1))
        hs['forward_2'][1] = forward_h.cpu()

        for i in range(2, T):
            spatial_h = hs['spatial'][i].cuda()
            flow_n1 = forward_flows[i - 1].cuda()
            forward_h_n1 = forward_h
            aligned_forward_h_n1, _ = warp(forward_h_n1, flow_n1)
            flow_n2 = flow_n1 + warp(forward_flows[i - 2].cuda(), flow_n1)[0]
            forward_h_n2 = hs['forward_2'][i - 2].cuda()
            aligned_forward_h_n2, _ = warp(forward_h_n2, flow_n2)
            feat_prop = self.deform_align['forward_2'](torch.cat((forward_h_n1, forward_h_n2), dim=1), torch.cat(
                (aligned_forward_h_n1, spatial_h, aligned_forward_h_n2), dim=1), flow_n1, flow_n2)
            forward_h = self.backbone['forward_2'](
                torch.cat((spatial_h, feat_prop, hs['forward_1'][i].cuda(), hs['backward_1'][i].cuda()), dim=1))
            hs['forward_2'][i] = forward_h.cpu()

        # extract backward features
        spatial_h = hs['spatial'][-1].cuda()
        backward_h = self.backbone['backward_2'](
            torch.cat((spatial_h, zeros_h.cuda(), hs['forward_1'][-1].cuda(), hs['backward_1'][-1].cuda(), hs['forward_2'][-1].cuda()),
                      dim=1))
        hs['backward_2'][-1] = backward_h.cpu()

        spatial_h = hs['spatial'][-2].cuda()
        flow_p1 = backward_flows[-1].cuda()
        backward_h_p1 = backward_h
        aligned_backward_h_p1, _ = warp(backward_h_p1, flow_p1)
        feat_prop = self.deform_align['backward_2'](torch.cat((backward_h_p1, zeros_h.cuda()), dim=1),
                                                    torch.cat((aligned_backward_h_p1, spatial_h, zeros_h.cuda()), dim=1),
                                                    flow_p1, zeros_flow.cuda())
        backward_h = self.backbone['backward_2'](
            torch.cat((spatial_h, feat_prop, hs['forward_1'][-2].cuda(), hs['backward_1'][-2].cuda(), hs['forward_2'][-2].cuda()),
                      dim=1))
        hs['backward_2'][-2] = backward_h.cpu()

        for i in range(3, T + 1):
            spatial_h = hs['spatial'][T - i].cuda()
            flow_p1 = backward_flows[T - i].cuda()
            backward_h_p1 = backward_h
            aligned_backward_h_p1, _ = warp(backward_h_p1, flow_p1)
            flow_p2 = flow_p1 + warp(backward_flows[T - i + 1].cuda(), flow_p1)[0]
            backward_h_p2 = hs['backward_2'][T - i + 1].cuda()
            aligned_backward_h_p2, _ = warp(backward_h_p2, flow_p2)
            feat_prop = self.deform_align['backward_2'](torch.cat((backward_h_p1, backward_h_p2), dim=1),
                                                        torch.cat((aligned_backward_h_p1, spatial_h, backward_h_p2),
                                                                  dim=1), flow_p1, flow_p2)
            backward_h = self.backbone['backward_2'](torch.cat((spatial_h, feat_prop, hs['forward_1'][T - i].cuda(),
                                                                hs['backward_1'][T - i].cuda(), hs['forward_2'][T - i].cuda()),
                                                               dim=1))
            hs['backward_2'][T - i] = backward_h.cpu()

        # generate results
        for i in range(T):
            seqdn[:, i] = self.d(
                torch.cat((hs['forward_1'][i].cuda(), hs['backward_1'][i].cuda(), hs['forward_2'][i].cuda(), hs['backward_2'][i].cuda()), dim=1)).cpu()

        return seqdn


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)