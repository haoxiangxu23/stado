from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch import nn

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0

        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'], 1),
        )
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'], 1)
        )
        fill_fc_weights(self.mov)

        wh_head_conv = 64 if arch == 'resnet' else head_conv
        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K, 1)
        )
        fill_fc_weights(self.wh)

        # --------------------- added for STADO -----------------------
        mgan_head_conv = 512
        self.mgan = nn.Sequential(
            nn.Conv2d(input_channel, mgan_head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mgan_head_conv, mgan_head_conv, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mgan_head_conv, branch_info['mgan'] // K, 1),
            nn.Sigmoid(),
        )
        fill_fc_weights(self.mgan)
        # -------------------------------------------------------------

    def forward(self, input_chunk):
        output = {}

        output_mgan = []
        for k, feature in enumerate(input_chunk):
            output_mgan.append(self.mgan(feature))
            input_chunk[k] = feature * output_mgan[-1].expand_as(feature) # (B, 1, H, W) expand to (B, input_channel, H, W)
        output['mgan'] = torch.cat(output_mgan, dim=1)

        output_wh = []
        for feature in input_chunk:
            output_wh.append(self.wh(feature))
        output['wh'] = torch.cat(output_wh, dim=1)

        input_chunk = torch.cat(input_chunk, dim=1)
        output['hm'] = self.hm(input_chunk)
        output['mov'] = self.mov(input_chunk)
        return output

