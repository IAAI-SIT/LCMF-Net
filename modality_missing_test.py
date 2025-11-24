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
from SSMTrain.ModalityMissingDataset import Dataset as CrossDataset
# from hausdorff import hausdorff_distance
from medpy.metric.binary import hd95
import my_model.RI_network3 as WNetCross
from tools.metrics import dice_coef, batch_iou, mean_iou, iou_score, ppv, sensitivity
import imageio


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='GLI-Model',
                        help='model name')
    parser.add_argument('--mode', default='GetPicture',
                        help='GetPicture or Calculate')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use for training')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    args = parser.parse_args()

    return args



def main():
    val_args = parse_args()
    torch.manual_seed(21)
    torch.cuda.manual_seed_all(21)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    args = joblib.load('/models/%s/args.pkl' % val_args.name)
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

    joblib.dump(args, '/models/%s/args.pkl' % args.name)

    # create model
    print("=> creating model %s" % args.arch)
    model = WNetCross.__dict__[args.arch](args, in_chanel=5)


    model = model.cuda()

    checkpoint = torch.load('/models/%s/model.pth' % args.name, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # Data loading code
    val_img_paths = sorted(glob(r'/Dataset/ModalityMissing/t1/*'))
    val_t1img_paths = '/Dataset/ModalityMissing/t1/'
    val_t1ceimg_paths = '/Dataset/ModalityMissing/t1ce/'
    val_t2img_paths = '/Dataset/ModalityMissing/t2/'
    val_mask_paths = sorted(glob(r'/Dataset/ModalityMissing/Mask/*'))



    val_dataset = CrossDataset(args, val_img_paths, val_mask_paths, t1='',t1ce=val_t1ceimg_paths,t2=val_t2img_paths,slice_num=2,FLAIR='t1')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_args.batch_size,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        num_workers=args.num_workers,
        drop_last=False)
    prediction_path = '/output/%s/' % args.name + "prediction_ModalityMissing_202506/"
    label_path = '/output/%s/' % args.name + "label_ModalityMissing_202506/"
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    if val_args.mode == "GetPicture":


        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
                    flair_image, t1_image, t1ce_image = sample['flair_image'], sample['t1_image'],sample['t1ce_image']
                    t2_image,label = sample['t2_image'], sample['label']

                    flair_image, t1_image, t1ce_image = flair_image.to(device), t1_image.to(device), t1ce_image.to(device)
                    t2_image,label = t2_image.to(device), label.cpu().numpy()

                    output = model(flair_image, t1_image, t1ce_image, t2_image)
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[val_args.batch_size * i:val_args.batch_size * (i + 1)]
                    for i in range(output.shape[0]):
                        npName = os.path.basename(img_paths[i])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgbName = rgbName + ".png"
                        rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i, 0, idx, idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                if output[i, 1, idx, idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                if output[i, 2, idx, idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                        imsave('output/%s/' % args.name +"/prediction_ModalityMissing_202506/"+ rgbName, rgbPic)

            torch.cuda.empty_cache()

        print("Saving GT,numpy to picture")
        val_gt_path = 'output/%s/' % args.name + "GT_ModalityMissing_202506/"
        if not os.path.exists(val_gt_path):
            os.mkdir(val_gt_path)
        for idx in tqdm(range(len(val_mask_paths))):
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            rgbName = name + ".png"

            npmask = np.load(mask_path)

            GtColor = np.zeros([npmask.shape[0], npmask.shape[1], 3], dtype=np.uint8)
            for idx in range(npmask.shape[0]):
                for idy in range(npmask.shape[1]):

                    if npmask[idx, idy] == 1:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 0
                        GtColor[idx, idy, 2] = 0

                    elif npmask[idx, idy] == 2:
                        GtColor[idx, idy, 0] = 0
                        GtColor[idx, idy, 1] = 128
                        GtColor[idx, idy, 2] = 0

                    elif npmask[idx, idy] == 3:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 255
                        GtColor[idx, idy, 2] = 0

            # imsave(val_gt_path + rgbName, GtColor)
            imageio.imwrite(val_gt_path + rgbName, GtColor)

        print("Done!")

    if val_args.mode == "Calculate":

        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []

        wtMaskList = []
        tcMaskList = []
        etMaskList = []
        wtPbList = []
        tcPbList = []
        etPbList = []

        maskPath = glob("output/%s/" % args.name + "GT_ModalityMissing_202506/*.png")
        pbPath = glob("output/%s/" % args.name +"prediction_ModalityMissing_202506/*.png")
        if len(maskPath) == 0:
            print("please GetPicture first!")
            return
        dice_batter = 0.8

        for myi in tqdm(range(len(maskPath))):
            mask = imread(maskPath[myi])
            pb = imread(pbPath[myi])
            better_name = os.path.basename(maskPath[myi])

            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
                    if mask[idx, idy, :].any() != 0:
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy, :].any() != 0:
                        wtpbregion[idx, idy] = 1
                    if mask[idx, idy, 0] == 255:
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy, 0] == 255:
                        tcpbregion[idx, idy] = 1
                    if mask[idx, idy, 1] == 255:
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy, 1] == 255:
                        etpbregion[idx, idy] = 1
            all_dice = 0
            dice = dice_coef(wtpbregion, wtmaskregion)
            all_dice += dice
            wt_dices.append(dice)
            ppv_n = ppv(wtpbregion, wtmaskregion)
            wt_ppvs.append(ppv_n)
            if  np.all(wtpbregion == 0) or np.all(wtmaskregion == 0):
                Hausdorff = 1
            else:
                Hausdorff = hd95(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
            wt_sensitivities.append(sensitivity_n)
            dice = dice_coef(tcpbregion, tcmaskregion)
            tc_dices.append(dice)
            all_dice += dice
            ppv_n = ppv(tcpbregion, tcmaskregion)
            tc_ppvs.append(ppv_n)
            if  np.all(tcpbregion == 0) or np.all(tcmaskregion == 0):
                Hausdorff = 1
            else:
                Hausdorff = hd95(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
            tc_sensitivities.append(sensitivity_n)
            dice = dice_coef(etpbregion, etmaskregion)
            et_dices.append(dice)
            all_dice += dice
            ppv_n = ppv(etpbregion, etmaskregion)
            et_ppvs.append(ppv_n)
            if  np.all(etpbregion == 0) or np.all(etmaskregion == 0):
                Hausdorff = 1
            else:
                Hausdorff = hd95(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(etpbregion, etmaskregion)
            et_sensitivities.append(sensitivity_n)

        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("=============")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("=============")
        print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
        print("=============")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")


if __name__ == '__main__':
    main()
