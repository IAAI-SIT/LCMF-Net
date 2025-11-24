# -*- coding: utf-8 -*-
import time
import os
import numpy as np
import argparse
from glob import glob
from collections import OrderedDict
import joblib
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from Brats2023DataSet import Dataset as CrossDataset
from tools.metrics import dice_coef, batch_iou, mean_iou, iou_score
from tools.utils import str2bool, count_params
import pandas as pd
import RI_network3 as WNetCross
from tools import losses
import random
import torch.nn.functional as F

arch_names = list(WNetCross.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir='/runs/GLI-Model')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='GLI-Model',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Mmodel',
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="Brats2021",
                        help='dataset name')
    parser.add_argument('--input-channels', default=5, type=int,
                        help='input channels')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='data -loder num_workers')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=True, type=str2bool)
    parser.add_argument('--loss', default='ComboLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=40, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--gpu', default='e',type=str,help='GPu id to use for training')
    args = parser.parse_args()


    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_for_nan(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print("Tensor contains NaN of Inf values")
        return True
    return False

def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def train(args, train_loader, model, criterion, optimizer):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    losses = AverageMeter()
    dices = AverageMeter()
    model.train()
    criterion = criterion.to(device)
    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):

        flair_image, t1_image, t1ce_image = sample['flair_image'], sample['t1_image'],sample['t1ce_image']
        t2_image,label = sample['t2_image'], sample['label']

        flair_image, t1_image, t1ce_image = flair_image.to(device), t1_image.to(device), t1ce_image.to(device)
        t2_image,label = t2_image.to(device), label.to(device)

        output = model(flair_image, t1_image, t1ce_image, t2_image)
        loss = criterion(output, label)

        dice = dice_coef(label, output.sigmoid().round())

        losses.update(loss.item(), flair_image.size(0))
        dices.update(dice.item(), flair_image.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('dice', dices.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    losses = AverageMeter()
    dices = AverageMeter()
    criterion = criterion.to(device)
    model.eval()

    with torch.no_grad():
       for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):

            flair_image, t1_image, t1ce_image = sample['flair_image'], sample['t1_image'],sample['t1ce_image']
            t2_image,label = sample['t2_image'], sample['label']

            flair_image, t1_image, t1ce_image = flair_image.to(device), t1_image.to(device), t1ce_image.to(device)
            t2_image,label = t2_image.to(device), label.to(device)

            output = model(flair_image, t1_image, t1ce_image, t2_image)
            loss = criterion(output, label)
            dice = dice_coef(label, output.sigmoid().round())


            losses.update(loss.item(), flair_image.size(0))
            dices.update(dice.item(), flair_image.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('dice', dices.avg),
    ])

    return log

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(img_paths,t1img_paths,t2img_paths,mask_paths):
    set_seed(21)
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    # args.dataset = "datasets"
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_170_train' % (args.dataset, args.arch)
        else:
            args.name = '%s_%s_170_train' % (args.dataset, args.arch)
    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.loss == 'DiceLoss':
        criterion = losses.__dict__[args.loss](3).cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()
    cudnn.benchmark = True
    # Data loading code

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.15, random_state=41)
    print("train_num:%s" % str(len(train_img_paths)))
    print("val_num:%s" % str(len(val_img_paths)))

    # create model
    print("=> creating model %s" % args.arch)
    model = WNetCross.__dict__[args.arch](args, in_chanel=5)

    model = model.to(device)

    print(count_params(model))
    optimizer = None
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = CrossDataset(args, img_paths=train_img_paths, mask_paths=train_mask_paths,t1=t1img_paths,t1ce=t1ceimg_paths,t2=t2img_paths,slice_num=2, aug=args.aug)
    val_dataset = CrossDataset(args, img_paths=val_img_paths, mask_paths=val_mask_paths,t1=t1img_paths,t1ce=t1ceimg_paths,t2=t2img_paths,slice_num=2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        num_workers=args.num_workers,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        num_workers=args.num_workers,
        drop_last=True)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'dice', 'val_loss', 'val_dice'
    ])

    best_dice = 0
    start_epoch = 0
    checkpoint_path = 'models/%s/model.pth' % args.name
    if os.path.exists(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['dice']
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))

    trigger = 0
    for epoch in range(start_epoch, args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))
        epoch_start_time = time.time()
        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss %.4f - dice %.4f - val_loss %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))
        writer.add_scalar('Loss/train', train_log['loss'], epoch)
        writer.add_scalar('Loss/val', val_log['loss'], epoch)
        writer.add_scalar('dice/train', train_log['dice'], epoch)
        writer.add_scalar('dice/val', val_log['dice'], epoch)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_hours, epoch_rest = divmod(epoch_time, 3600)
        epoch_minutes, epoch_seconds = divmod(epoch_rest, 60)
        print(f"epoch training time：{int(epoch_hours)} h {int(epoch_minutes)} m {int(epoch_seconds)} s")
        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['dice'],
            val_log['loss'],
            val_log['dice'],
        ], index=['epoch', 'lr', 'loss', 'dice', 'val_loss', 'val_dice'])

        log = log._append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' % args.name, index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'dice': val_log['dice'],
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, 'models/%s/model.pth' % args.name)
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
    writer.close()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    start_time = time.time()
    dataPath = '/Dataset/Brats2023GLI'
    img_paths = glob(r'/Dataset/Brats2023GLI/flair/*')
    t1img_paths = '/Dataset/Brats2023GLI/t1/'
    t1ceimg_paths = '/Dataset/Brats2023GLI/t1ce/'
    t2img_paths = '/Dataset/Brats2023GLI/t2/'
    mask_paths = glob(r'/Dataset/Brats2023GLI/Mask/*')
    main(img_paths,t1img_paths,t2img_paths,mask_paths)
    end_time = time.time()
    training_duration = end_time - start_time
    hours, rest = divmod(training_duration, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"training time：{int(hours)} h {int(minutes)} m {int(seconds)} s")
