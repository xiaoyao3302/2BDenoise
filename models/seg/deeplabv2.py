# donot use deep supervision in JCAS
from models.backbone.resnet_deeplab_v2 import ResNet_pretrained, ASPP_module

import torch
from torch import nn
import torch.nn.functional as F
import pdb


class DeepLabV2(nn.Module):
    def __init__(self, args):
        super(DeepLabV2, self).__init__()

        # backbone (encoder)
        self.encoder = DeepLabV2_Encoder(args)

        self.cls = DeepLabV2_cls(args)
        
    def forward(self, x):

        feature = self.encoder(x)
        pred = self.cls(x, feature)

        return feature, pred


class DeepLabV2_Encoder(nn.Module):
    def __init__(self, args):
        super(DeepLabV2_Encoder, self).__init__()

        # introduce network perturbations
        self.num_class = args.num_class

        # backbone (encoder)
        self.backbone = ResNet_pretrained(args)
        
    def forward(self, x):

        # backbone encoder
        feats = self.backbone.forward(x)
        feature = feats[-1]

        return feature


class DeepLabV2_cls(nn.Module):
    def __init__(self, args):
        super(DeepLabV2_cls, self).__init__()

        # introduce network perturbations
        self.num_class = args.num_class

        self.decoder = self._make_pred_layer(ASPP_module, 2048, [6, 12, 18], [6, 12, 18], self.num_class)
        
    def forward(self, x, feature):
        h, w = x.shape[-2:]

        # decoder

        pred = self._decode(feature)

        interp_func = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        pred = interp_func(pred)

        return pred

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_class):
        return block(inplanes, dilation_series, padding_series, num_class)

    def _decode(self, feat):

        pred = self.decoder(feat)

        return pred
        
