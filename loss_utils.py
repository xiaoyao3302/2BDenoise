import torch
import torch.nn as nn
import numpy as np
import random
import pdb
import torch.nn.functional as F
from itertools import filterfalse
from torch.nn.functional import softmax
import argparse
import pdb


class FocalLoss(torch.nn.Module):
    # adopted from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/focal_loss.py
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
    """

    def __init__(self, args):
        super(FocalLoss, self).__init__()
        self.args = args
        self.num_class = args.num_class
        self.alpha = args.Focal_alpha
        self.gamma = args.Focal_gamma
        self.smooth = 1e-4
        self.device = args.device

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class).to(self.device)
        elif isinstance(self.alpha, (int, float)):
            self.alpha = torch.as_tensor([self.alpha] * self.num_class).to(self.device)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(self.alpha).to(self.device)
        if self.alpha.shape[0] != self.num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, pred, labels):
        B, C, H, W = pred.shape
        
        pred_p = F.softmax(pred, dim=1)

        if pred_p.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            pred_p = pred_p.view(B, C, H*W)
            pred_p = pred_p.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            pred_p = pred_p.view(-1, C)  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        labels = labels.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # ----------memory saving way--------
        pred_p = pred_p.gather(1, labels).view(-1) + self.smooth  # avoid nan
        log_pred_p = torch.log(pred_p)

        alpha_class = self.alpha[labels.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, pred_p), self.gamma)
        loss = class_weight * log_pred_p

        loss = loss.mean()
        return loss
    

class SCELoss(torch.nn.Module):
    def __init__(self, args):
        super(SCELoss, self).__init__()
        self.alpha = args.SCE_alpha
        self.beta = args.SCE_beta
        self.num_class = args.num_class

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        device = pred.device
        # CE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_class).float().to(device).permute(0, 3, 1, 2)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()

        logits = {}
        logits['ce'] = ce
        logits['rce'] = rce.mean()
        logits['total'] = loss
        return logits
    

class DiceLoss(nn.Module):
    def __init__(self, args):
        super(DiceLoss, self).__init__()
        self.args = args
        self.num_class = args.num_class

    def forward(self, pred, labels):
        # pred and labels: B
        device = pred.device

        # pred: softmax
        pred = F.softmax(pred, dim=1)

        # label: one hot
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_class).float().to(device).permute(0, 3, 1, 2)  # B, num_class
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        intersection = (pred * label_one_hot).sum()                            
        dice = (2. * intersection + 1e-6)/(pred.sum() + label_one_hot.sum() + 1e-6)  
        
        dice = 1 - dice

        logits = {}
        logits['dice'] = dice
        logits['total'] = dice
        return logits


class DiceLoss_v2(nn.Module):
    def __init__(self, args):
        super(DiceLoss, self).__init__()
        self.num_class = args.num_class

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_class):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
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

    def forward(self, inputs, target):
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.num_class):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice

        loss = loss / self.num_class

        logits = {}
        logits['dice'] = loss
        logits['total'] = loss
        return logits
    

class Dice_CELoss(nn.Module):
    def __init__(self, args):
        super(Dice_CELoss, self).__init__()
        self.args = args
        self.num_class = args.num_class
        self.alpha = args.Dice_CE_alpha
        self.beta = args.Dice_CE_beta
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # pred and labels: B
        device = pred.device

        # CE
        ce_loss = self.cross_entropy(pred, labels)

        # Dice
        # pred: softmax
        pred = F.softmax(pred, dim=1)

        # label: one hot
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_class).float().to(device).permute(0, 3, 1, 2)  # B, num_class, H, W
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        intersection = (pred * label_one_hot).sum()                            
        dice = (2. * intersection + 1e-6)/(pred.sum() + label_one_hot.sum() + 1e-6)  
        dice_loss = 1 - dice

        total_loss = self.alpha * ce_loss + self.beta * dice_loss
        
        logits = {}
        logits['ce'] = ce_loss
        logits['dice'] = dice_loss
        logits['total'] = total_loss
        return logits


class Dice_CELoss_v2(nn.Module):
    def __init__(self, args):
        super(DiceLoss, self).__init__()
        self.num_class = args.num_class
        self.alpha = args.Dice_CE_alpha
        self.beta = args.Dice_CE_beta
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_class):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
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

    def forward(self, inputs, target):
        # CE
        ce_loss = self.cross_entropy(inputs, target)

        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        dice_loss = 0.0
        for i in range(0, self.num_class):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            dice_loss += dice
        dice_loss = dice_loss / self.num_class
        
        total_loss = self.alpha * ce_loss + self.beta * dice_loss

        logits = {}
        logits['ce'] = ce_loss
        logits['dice'] = dice_loss
        logits['total'] = total_loss

        return logits
    

class NR_DiceLoss(nn.Module):
    def __init__(self, args):
        super(NR_DiceLoss, self).__init__()
        self.args = args
        self.num_class = args.num_class
        self.gamma = args.NR_Dice_gamma

    def forward(self, pred, labels):
        # pred and labels: B
        device = pred.device

        # pred: softmax
        pred = F.softmax(pred, dim=1)

        # label: one hot
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_class).float().to(device).permute(0, 3, 1, 2)  # B, num_class
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        NR_Dice = (torch.abs(pred - label_one_hot) ** self.gamma).sum() / ((pred ** 2).sum() + (label_one_hot ** 2).sum() + 1e-6)  
        
        logits = {}
        logits['NR_dice'] = NR_Dice
        logits['total'] = NR_Dice
        return logits
    

class NR_Dice_CELoss(nn.Module):
    def __init__(self, args):
        super(NR_Dice_CELoss, self).__init__()
        self.args = args
        self.num_class = args.num_class
        self.alpha = args.NR_Dice_CE_alpha
        self.beta = args.NR_Dice_CE_beta
        self.gamma = args.NR_Dice_CE_gamma

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # pred and labels: B
        device = pred.device

        # CE
        ce_loss = self.cross_entropy(pred, labels)

        # pred: softmax
        pred = F.softmax(pred, dim=1)

        # label: one hot
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_class).float().to(device).permute(0, 3, 1, 2)  # B, num_class
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        NR_Dice = (torch.abs(pred - label_one_hot) ** self.gamma).sum() / ((pred ** 2).sum() + (label_one_hot ** 2).sum() + 1e-6)  

        total_loss = self.alpha * ce_loss + self.beta * NR_Dice

        logits = {}
        logits['ce'] = ce_loss
        logits['NR_dice'] = NR_Dice
        logits['total'] = total_loss
        return logits
    

class IoULoss(nn.Module):
    def __init__(self, args):
        super(IoULoss, self).__init__()
        self.args = args
        self.num_class = args.num_class

    def forward(self, pred, labels):
        # pred and labels: B
        device = pred.device

        # pred: softmax
        pred = F.softmax(pred, dim=1)

        # label: one hot
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_class).float().to(device).permute(0, 3, 1, 2)  # B, num_class
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        intersection = (pred * label_one_hot).sum()   
        total = pred.sum() + label_one_hot.sum()
        union = total - intersection 
        
        IoU = (intersection + 1e-6)/(union + 1e-6)
                
        logits = {}
        logits['IoU'] = IoU
        logits['total'] = IoU
        return logits
    

class IoU_CELoss(nn.Module):
    def __init__(self, args):
        super(IoU_CELoss, self).__init__()
        self.args = args
        self.num_class = args.num_class
        self.alpha = args.IoU_CE_alpha
        self.beta = args.IoU_CE_beta

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # pred and labels: B
        device = pred.device

        # CE
        ce_loss = self.cross_entropy(pred, labels)

        # pred: softmax
        pred = F.softmax(pred, dim=1)

        # label: one hot
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_class).float().to(device).permute(0, 3, 1, 2)  # B, num_class
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        intersection = (pred * label_one_hot).sum()   
        total = pred.sum() + label_one_hot.sum()
        union = total - intersection 
        
        IoU = (intersection + 1e-6)/(union + 1e-6)

        IoU_loss = 1 - IoU

        total_loss = self.alpha * ce_loss + self.beta * IoU_loss
                
        logits = {}
        logits['ce'] = ce_loss
        logits['IoU'] = IoU_loss
        logits['total'] = total_loss
        return logits
    

class ComboLoss(nn.Module):
    def __init__(self, args):
        super(ComboLoss, self).__init__()
        self.args = args
        self.num_class = args.num_class
        self.alpha = args.Combo_alpha
        self.ce_ratio = args.Combo_ce_ratio

    def forward(self, pred, labels):
        # pred and labels: B
        device = pred.device

        # pred: softmax
        pred = F.softmax(pred, dim=1)

        # label: one hot
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_class).float().to(device).permute(0, 3, 1, 2)  # B, num_class
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        intersection = (pred * label_one_hot).sum()                            
        dice = (2. * intersection + 1e-6)/(pred.sum() + label_one_hot.sum() + 1e-6)  

        pred = torch.clamp(pred, 1e-9, 1.0 - 1e-9)    
        out = - (self.alpha * ((label_one_hot * torch.log(pred)) + ((1 - self.alpha) * (1.0 - label_one_hot) * torch.log(1.0 - pred))))
        weighted_ce = out.mean()

        combo = (self.ce_ratio * weighted_ce) + ((1 - self.ce_ratio) * dice)
        
        logits = {}
        logits['weighted_ce'] = weighted_ce
        logits['dice'] = dice
        logits['total'] = combo
        return logits


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.eps = torch.as_tensor(1e-10)
        self.num_class = args.num_class

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-class Lovasz-Softmax loss. Adapted from github.com/bermanmaxim/LovaszSoftmax

        :param prediction: NCHW tensor, raw logits from the network
        :param target: NHW tensor, ground truth labels
        :return: Lovász-Softmax loss
        """
        p = softmax(prediction, dim=1)
        loss = self.lovasz_softmax_flat(*self.flatten_probabilities(p, target))
        return loss

    def lovasz_softmax_flat(self, prob: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
        """Multi-class Lovasz-Softmax loss. Adapted from github.com/bermanmaxim/LovaszSoftmax

        :param prob: class probabilities at each prediction (between 0 and 1)
        :param lbl: ground truth labels (between 0 and C - 1)
        :return: Lovász-Softmax loss
        """
        if prob.numel() == 0:
            # only void pixels, the gradients should be 0
            return prob * 0.
        c = prob.size(1)
        losses = []
        class_to_sum = list(range(c))

        for c in class_to_sum:
            fg = (lbl == c).float()  # foreground for class c
            class_pred = prob[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.detach()
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
        return losses

    def flatten_probabilities(self, prob: torch.Tensor, lbl: torch.Tensor):
        """
        Flattens predictions in the batch
        """
        if prob.dim() == 3:
            # assumes output of a sigmoid layer
            n, h, w = prob.size()
            prob = prob.view(n, 1, h, w)
        _, c, _, _ = prob.size()
        prob = prob.permute(0, 2, 3, 1).contiguous().view(-1, c)  # B * H * W, C = P, C
        lbl = lbl.view(-1)
        return prob, lbl


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(ip: torch.Tensor, ignore_nan: bool = False, empty=0):
    """
    nanmean compatible with generators.
    """
    ip = iter(ip)
    if ignore_nan:
        ip = filterfalse(isnan, ip)
    try:
        n = 1
        acc = next(ip)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(ip, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='loss')
    
    parser.add_argument('--num_class', type=int, default=3, help='number of classes per dataset')
    parser.add_argument('--Focal_alpha', type=float, default=[1, 3, 5], help='Focal_alpha')
    parser.add_argument('--Focal_gamma', type=float, default=2.0, help='Focal_gamma')
    parser.add_argument('--SCE_alpha', type=float, default=0.5, help='SCE_alpha')
    parser.add_argument('--SCE_beta', type=float, default=0.5, help='SCE_beta')
    parser.add_argument('--NR_Dice_gamma', type=float, default=1.5, help='NR_Dice_gamma, between 1 and 2')
    parser.add_argument('--Combo_alpha', type=float, default=0.5, help='Combo_alpha')
    parser.add_argument('--Combo_ce_ratio', type=float, default=0.5, help='Combo_ce_ratio')

    args = parser.parse_args()

    pred = torch.rand([4, 3, 5, 5])
    label = torch.rand([4, 5, 5]) / 0.5
    label = label.round().long()

    loss_Focal = FocalLoss(args)
    loss_Dice = DiceLoss(args)
    loss_IoU = IoULoss(args)
    loss_SCE = SCELoss(args)
    loss_NR_Dice = NR_DiceLoss(args)
    loss_Combo = ComboLoss(args)
    loss_LovaszSoftmax = LovaszSoftmaxLoss(args)

    focal_result = loss_Focal(pred, label)
    dice_result = loss_Dice(pred, label)
    IoU_result = loss_IoU(pred, label)
    SCE_result = loss_SCE(pred, label)
    NR_Dice_result = loss_NR_Dice(pred, label)
    Combo_result = loss_Combo(pred, label)
    LovaszSoftmax_result = loss_LovaszSoftmax(pred, label)

    print('focal_result:', focal_result)
    print('dice_result:', dice_result)
    print('IoU_result:', IoU_result)
    print('SCE_result:', SCE_result)
    print('NR_Dice_result:', NR_Dice_result)
    print('Combo_result:', Combo_result)
    print('LovaszSoftmax_result:', LovaszSoftmax_result)