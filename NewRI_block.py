import torch
import torch.nn as nn
from attention.SCSA import SCSA

class IncResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(IncResBlock, self).__init__()
        self.SCSA_cfg = dict(
            attn_drop_ratio=0.0,
            down_sample_mode='avg_pool',
            gate_layer='sigmoid',
            group_kernel_sizes=[
                3,
                5,
                7,
                9,
            ],
            head_num=1,
            window_size=7)
        self.atten = SCSA
        self.Inputconv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        #
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4))
        # nn.ReLU())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.relu = nn.ReLU()
        self.stride = stride
        self.pool = nn.MaxPool2d(2)
        self.atten = SCSA(planes, **self.SCSA_cfg)

    def forward(self, x):
        residual = self.Inputconv1x1(x)

        c1 = self.conv1_1(x)
        c2 = self.conv1_2(x)
        c3 = self.conv1_3(x)
        c4 = self.conv1_4(x)

        out = torch.cat([c1, c2, c3, c4], 1)

        # adding the skip connection
        out += residual
        skip = self.relu(out)
        skip = self.atten(skip)
        out = self.pool(skip)

        return out, skip


class IncResUpBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(IncResUpBlock, self).__init__()
        self.Inputconv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        #
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4))
        # nn.ReLU())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.relu = nn.ReLU()
        self.stride = stride
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip_input=None):
        x = self.up_sample(x)
        if skip_input is not None:
            x = torch.cat([x, skip_input], dim=1)
        residual = self.Inputconv1x1(x)

        c1 = self.conv1_1(x)
        c2 = self.conv1_2(x)
        c3 = self.conv1_3(x)
        c4 = self.conv1_4(x)

        out = torch.cat([c1, c2, c3, c4], 1)

        # adding the skip connection
        out += residual
        out = self.relu(out)

        return out
