import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-6
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.mean()
        loss = 0.5 * bce + dice
        return loss


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input, target):
        # Focal Loss part
        input = torch.sigmoid(input)
        bce = F.binary_cross_entropy(input, target, reduction='none')
        bce_exp = torch.exp(-bce)
        focal_loss = (1 - bce_exp) ** self.gamma * bce
        focal_loss = focal_loss.mean()
        
        # Dice Loss part
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        # 获取类别数
        num_classes = inputs.size(1)
        
        # 将预测结果和真实标签展平
        inputs = inputs.contiguous().view(-1, num_classes)
        targets = targets.contiguous().view(-1, num_classes)
        
        # 计算类别权重
        w = 1.0 / (torch.sum(targets, dim=0) ** 2 + smooth)
        
        # 计算分子和分母
        intersection = torch.sum(inputs * targets, dim=0)
        union = torch.sum(inputs + targets, dim=0)
        
        # 计算 Generalized Dice Loss
        gdl = 1 - 2 * torch.sum(w * intersection) / (torch.sum(w * union) + smooth)
        
        return gdl
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.sigmoid(inputs)
        # target = self._one_hot_encoder(target)#[12, 6, 256, 256]
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
