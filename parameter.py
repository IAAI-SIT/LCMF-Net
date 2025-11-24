# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import joblib
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from Brats2023DataSet import Dataset as CrossDataset
# from hausdorff import hausdorff_distance
from medpy.metric.binary import hd95
import my_model.bottom_add_model as WNetCross


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='MEN-Model',
                        help='model name')
    parser.add_argument('--mode', default='GetPicture',
                        help='GetPicture or Calculate')
    parser.add_argument('--gpu', default='1', type=str, help='GPU id to use for training')
    parser.add_argument('-b', '--batch-size', default=18, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    args = parser.parse_args()

    return args


from torchinfo import summary
import torch

def print_model_info(model, input_sizes):
    summary(model, input_data=tuple(torch.randn(size).to(next(model.parameters()).device) for size in input_sizes))


if __name__ == '__main__':
    val_args = parse_args()
    torch.manual_seed(21)  
    torch.cuda.manual_seed_all(21) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    args = joblib.load('/home/image/nvme/zJuny/SegMamba-main/SegMamba-main/models/%s/args.pkl' % val_args.name)
    device = torch.device(f"cuda:{val_args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    # os.environ['CUDA_VISIBLE_DEVICES'] = val_args.gpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('output/%s' % args.name):
        os.makedirs('output/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, '/home/image/nvme/zJuny/SegMamba-main/SegMamba-main/models/%s/args.pkl' % args.name)

    # create model
    print("=> creating model %s" % args.arch)
    model = WNetCross.__dict__[args.arch](args, in_chanel=5)
    model = model.cuda()
    model.eval()
    input_sizes = [(1, 5, 160, 160),  # flair
                   (1, 5, 160, 160),  # t1
                   (1, 5, 160, 160),  # t1ce
                   (1, 5, 160, 160)]  # t2
    print_model_info(model, input_sizes)