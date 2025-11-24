# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch.nn.functional as F
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair
from scipy import ndimage


# Non-Local Block for multi-cross attention
class NLBlockND_multicross_block(nn.Module):
    """
    Non-Local Block for multi-cross attention.

    Args:
        in_channels (int): Number of input channels.
        inter_channels (int, optional): Number of intermediate channels. Defaults to None.

    Attributes:
        in_channels (int): Number of input channels.
        inter_channels (int): Number of intermediate channels.
        g (nn.Conv2d): Convolutional layer for the 'g' branch.
        final (nn.Conv2d): Final convolutional layer.
        W_z (nn.Sequential): Sequential block containing a convolutional layer followed by batch normalization for weight 'z'.
        theta (nn.Conv2d): Convolutional layer for the 'theta' branch.
        phi (nn.Conv2d): Convolutional layer for the 'phi' branch.

    Methods:
        forward(x_thisBranch, x_otherBranch): Forward pass of the non-local block.

    """

    def __init__(self, in_channels, inter_channels=None):
        super(NLBlockND_multicross_block, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.final = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels)
        )

        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

    def forward(self, x_thisBranch, x_otherBranch):
        batch_size = x_thisBranch.size(0)
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)    # 1x1 Conv
        g_x = g_x.permute(0, 2, 1)  # C X H X W --> C X W X H

        theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1) # 1x1 Conv
        phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(phi_x, theta_x)    # Matrix Multiplication
        f_div_C = torch.sigmoid(f)  # 使用 Sigmoid 激活函数

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        z = self.W_z(y)
        return z


class NLBlockND_multicross(nn.Module):

    def __init__(self, in_channels, inter_channels=None):
        super(NLBlockND_multicross, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.cross_attention = NLBlockND_multicross_block(in_channels=in_channels, inter_channels=64)

    def forward(self, x_thisBranch, x_otherBranch):
        outputs = []
        for i in range(8):
            cross_attention = NLBlockND_multicross_block(in_channels=self.in_channels, inter_channels=64)
            cross_attention = cross_attention.to('cuda')
            output = cross_attention(x_thisBranch, x_otherBranch)
            outputs.append(output)
        final_output = torch.cat(outputs, dim=1)
        # final_output = final_output + x_thisBranch #Changed
        return final_output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownCross(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

        # Embeddings module for constructing embeddings from patches and position embeddings
