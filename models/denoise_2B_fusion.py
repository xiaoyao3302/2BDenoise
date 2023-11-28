# fusion or not
# 
from models.seg.deeplabv2 import DeepLabV2_Encoder, DeepLabV2_cls
from models.seg.deeplabv3plus import DeepLabV3Plus_Encoder, DeepLabV3Plus_cls
from models.seg.unet import UNet_Encoder, UNet_cls
from models.seg.DMFNet import DMFNet, MFNet, DMFNet_cls

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV2_2B_Denoise_model(nn.Module):
    def __init__(self, args):
        super(DeepLabV2_2B_Denoise_model, self).__init__()
        self.args = args
        self.use_fusion = args.use_fusion
        self.fusion_mode = args.fusion_mode     # noisy or clean

        # clean
        self.clean_encoder = DeepLabV2_Encoder(args)
        self.clean_cls = DeepLabV2_cls(args)
        
        # noisy
        self.noisy_encoder = DeepLabV2_Encoder(args)
        self.noisy_cls = DeepLabV2_cls(args)

        self.fusion = nn.Sequential(nn.Conv2d(4096, 256, 1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(True),
                                nn.Conv2d(256, 2048, 1, bias=False),
                                nn.BatchNorm2d(2048),
                                nn.ReLU(True))
        
    def forward(self, x):
        logits = {}

        if self.use_fusion:
            if self.fusion_mode == 'clean':
                # noise stream
                noisy_feature = self.noisy_encoder(x)
                noisy_pred = self.noisy_cls(x, noisy_feature)

                logits['noisy_feature'] = noisy_feature
                logits['noisy_pred'] = noisy_pred

                # fuse noisy to clean
                clean_feature = self.clean_encoder(x)

                all_feature = torch.cat([clean_feature, noisy_feature], dim=1)
                all_feature = self.fusion(all_feature)
                clean_pred = self.clean_cls(x, all_feature)

                logits['feature'] = all_feature
                logits['pred'] = clean_pred

            else:
                # clean stream
                clean_feature = self.clean_encoder(x)
                clean_pred = self.clean_cls(x, clean_feature)

                logits['feature'] = clean_feature
                logits['pred'] = clean_pred

                # fuse clean to noisy
                noisy_feature = self.noisy_encoder(x)

                all_feature = torch.cat([clean_feature, noisy_feature], dim=1)
                all_feature = self.fusion(all_feature)
                noisy_pred = self.noisy_cls(x, all_feature)

                logits['noisy_feature'] = all_feature
                logits['noisy_pred'] = noisy_pred

        else:
            # clean stream
            clean_feature = self.clean_encoder(x)
            clean_pred = self.clean_cls(x, clean_feature)

            logits['feature'] = clean_feature
            logits['pred'] = clean_pred

            # noisy stream
            noisy_feature = self.noisy_encoder(x)
            noisy_pred = self.noisy_cls(x, noisy_feature)

            logits['noisy_feature'] = noisy_feature
            logits['noisy_pred'] = noisy_pred

        return logits
    

class DeepLabV3Plus_2B_Denoise_model(nn.Module):
    def __init__(self, args):
        super(DeepLabV3Plus_2B_Denoise_model, self).__init__()
        self.args = args
        self.use_fusion = args.use_fusion
        self.fusion_mode = args.fusion_mode     # noisy or clean

        # clean
        self.clean_encoder = DeepLabV3Plus_Encoder(args)
        self.clean_cls = DeepLabV3Plus_cls(args)

        # noisy
        self.noisy_encoder = DeepLabV3Plus_Encoder(args)
        self.noisy_cls = DeepLabV3Plus_cls(args)

        self.fusion = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(True),
                                nn.Conv2d(256, 256, 1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(True))
        
    def forward(self, x):
        logits = {}

        if self.use_fusion:
            if self.fusion_mode == 'clean':
                # noise stream
                noisy_feature = self.noisy_encoder(x)
                noisy_pred = self.noisy_cls(x, noisy_feature)

                logits['noisy_feature'] = noisy_feature
                logits['noisy_pred'] = noisy_pred

                # fuse noisy to clean
                clean_feature = self.clean_encoder(x)

                all_feature = torch.cat([clean_feature, noisy_feature], dim=1)
                all_feature = self.fusion(all_feature)
                clean_pred = self.clean_cls(x, all_feature)

                logits['feature'] = all_feature
                logits['pred'] = clean_pred

            else:
                # clean stream
                clean_feature = self.clean_encoder(x)
                clean_pred = self.clean_cls(x, clean_feature)

                logits['feature'] = clean_feature
                logits['pred'] = clean_pred

                # fuse clean to noisy
                noisy_feature = self.noisy_encoder(x)

                all_feature = torch.cat([clean_feature, noisy_feature], dim=1)
                all_feature = self.fusion(all_feature)
                noisy_pred = self.noisy_cls(x, all_feature)

                logits['noisy_feature'] = all_feature
                logits['noisy_pred'] = noisy_pred

        else:
            # clean stream
            clean_feature = self.clean_encoder(x)
            clean_pred = self.clean_cls(x, clean_feature)

            logits['feature'] = clean_feature
            logits['pred'] = clean_pred

            # noisy stream
            noisy_feature = self.noisy_encoder(x)
            noisy_pred = self.noisy_cls(x, noisy_feature)

            logits['noisy_feature'] = noisy_feature
            logits['noisy_pred'] = noisy_pred
            
        return logits


class UNet_2B_Denoise_model(nn.Module):
    def __init__(self, args):
        super(UNet_2B_Denoise_model, self).__init__()
        self.args = args
        self.use_fusion = args.use_fusion
        self.fusion_mode = args.fusion_mode     # noisy or clean

        # clean
        self.clean_encoder = UNet_Encoder(args)
        self.clean_cls = UNet_cls(args)

        # noisy
        self.noisy_encoder = UNet_Encoder(args)
        self.noisy_cls = UNet_cls(args)

        self.fusion = nn.Sequential(nn.Conv2d(128, 256, 1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(True),
                                nn.Conv2d(256, 64, 1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(True))
        
    def forward(self, x):
        logits = {}

        if self.use_fusion:
            if self.fusion_mode == 'clean':
                # noise stream
                noisy_feature = self.noisy_encoder(x)
                noisy_pred = self.noisy_cls(x, noisy_feature)

                logits['noisy_feature'] = noisy_feature
                logits['noisy_pred'] = noisy_pred

                # fuse noisy to clean
                clean_feature = self.clean_encoder(x)

                all_feature = torch.cat([clean_feature, noisy_feature], dim=1)
                all_feature = self.fusion(all_feature)
                clean_pred = self.clean_cls(x, all_feature)

                logits['feature'] = all_feature
                logits['pred'] = clean_pred

            else:
                # clean stream
                clean_feature = self.clean_encoder(x)
                clean_pred = self.clean_cls(x, clean_feature)

                logits['feature'] = clean_feature
                logits['pred'] = clean_pred

                # fuse clean to noisy
                noisy_feature = self.noisy_encoder(x)

                all_feature = torch.cat([clean_feature, noisy_feature], dim=1)
                all_feature = self.fusion(all_feature)
                noisy_pred = self.noisy_cls(x, all_feature)

                logits['noisy_feature'] = all_feature
                logits['noisy_pred'] = noisy_pred

        else:
            # clean stream
            clean_feature = self.clean_encoder(x)
            clean_pred = self.clean_cls(x, clean_feature)

            logits['feature'] = clean_feature
            logits['pred'] = clean_pred

            # noisy stream
            noisy_feature = self.noisy_encoder(x)
            noisy_pred = self.noisy_cls(x, noisy_feature)

            logits['noisy_feature'] = noisy_feature
            logits['noisy_pred'] = noisy_pred
            
        return logits