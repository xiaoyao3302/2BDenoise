from models.seg.deeplabv2 import DeepLabV2
from models.seg.deeplabv3plus import DeepLabV3Plus
from models.seg.unet import UNet
from models.seg.DMFNet import DMFNet, MFNet, DMFNet_cls

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV2_Baseline_model(nn.Module):
    def __init__(self, args):
        super(DeepLabV2_Baseline_model, self).__init__()
        self.args = args
        self.branch = DeepLabV2(args)

    def forward(self, x):
        logits = {}

        feature, pred = self.branch(x)
        
        logits['pred'] = pred
        logits['feature'] = feature     # downsampled
        
        return logits
    

class DeepLabV3Plus_Baseline_model(nn.Module):
    def __init__(self, args):
        super(DeepLabV3Plus_Baseline_model, self).__init__()
        self.args = args
        self.branch = DeepLabV3Plus(args)

    def forward(self, x):
        logits = {}

        feature, pred = self.branch(x)
        
        logits['pred'] = pred
        logits['feature'] = feature     # downsampled
        
        return logits


class UNet_Baseline_model(nn.Module):
    def __init__(self, args):
        super(UNet_Baseline_model, self).__init__()
        self.args = args
        self.branch = UNet(args)

    def forward(self, x):
        logits = {}

        feature, pred = self.branch(x)
        
        logits['pred'] = pred
        logits['feature'] = feature     # downsampled
        
        return logits


class MFNet_Baseline_model(nn.Module):
    def __init__(self, args):
        super(MFNet_Baseline_model, self).__init__()
        self.args = args
        self.encoder = MFNet(args)
        self.cls = DMFNet_cls(args)

    def forward(self, x):
        logits = {}

        feature = self.encoder(x)
        pred = self.cls(feature)
        
        logits['pred'] = pred
        logits['feature'] = feature 
        
        return logits


class DMFNet_Baseline_model(nn.Module):
    def __init__(self, args):
        super(DMFNet_Baseline_model, self).__init__()
        self.args = args
        self.encoder = DMFNet(args)
        self.cls = DMFNet_cls(args)

    def forward(self, x):
        logits = {}

        feature = self.encoder(x)
        pred = self.cls(feature)
        
        logits['pred'] = pred
        logits['feature'] = feature 
        
        return logits
    
    