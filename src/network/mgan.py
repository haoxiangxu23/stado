import torch.nn as nn


class MGAN(nn.Module):
    def __init__(self, in_channels, conv_out_channels=512):
        super(MGAN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels, conv_out_channels, 3, padding=1))
        self.convs.append(nn.Conv2d(conv_out_channels, conv_out_channels, 3, padding=1))
        self.conv_logits = nn.Conv2d(conv_out_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = x
        for conv in self.convs:
            x = conv(x)
            x = self.relu(x)
        out = self.conv_logits(x).sigmoid() * feat
        return out

