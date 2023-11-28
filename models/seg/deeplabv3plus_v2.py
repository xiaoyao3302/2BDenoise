# v2: output after ASPP as decoder
import models.backbone.resnet_deeplab_v3plus as resnet
import torch
from torch import nn
import torch.nn.functional as F
import pdb


set_multi_grid = False
set_replace_stride_with_dilation = [False, False, True]
set_dilations = [6, 12, 18]


class DeepLabV3Plus(nn.Module):
    def __init__(self, args):
        super(DeepLabV3Plus, self).__init__()

        self.encoder = DeepLabV3Plus_Encoder(args)
        self.cls = DeepLabV3Plus_cls(args)

    def forward(self, x):
        c1, c4 = self.encoder(x)
        pred = self.cls(x, c1, c4)

        return c4, pred


class DeepLabV3Plus_Encoder(nn.Module):
    def __init__(self, args):
        super(DeepLabV3Plus_Encoder, self).__init__()

        self.backbone = \
            resnet.__dict__[args.backbone](True, multi_grid=set_multi_grid,
                                                replace_stride_with_dilation=set_replace_stride_with_dilation)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, set_dilations)

    def forward(self, x):

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        c4 = self.head(c4)                  # D: high_channels // 8 = 256

        return c1, c4


class DeepLabV3Plus_cls(nn.Module):
    def __init__(self, args):
        super(DeepLabV3Plus_cls, self).__init__()

        low_channels = 256
        high_channels = 2048

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        
        self.classifier = nn.Conv2d(256, args.num_class, 1, bias=True)

    def forward(self, x, c1, c4):
        h, w = x.shape[-2:]

        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(c1)
        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)
        pred = self.classifier(feature)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)

        return pred



def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

