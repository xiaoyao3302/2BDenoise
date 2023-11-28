import torch
from torch import nn
import torch.nn.functional as F
import copy
import math

import pdb



class AttentionBlock(nn.Module):
    def __init__(self, args):
        super(AttentionBlock, self).__init__()
        self.args = args
        self.model = args.model

        if self.model == 'DeepLabv3p':
            num_patch = 1024
            hidden_dim = 256
            conv_kernel = 8
        else:
            # TODO
            num_patch = 16
            hidden_dim = 256
            conv_kernel = 8
        
        self.linearQ = nn.Linear(hidden_dim, hidden_dim)
        self.linearK = nn.Linear(hidden_dim, hidden_dim)
        self.linearV = nn.Linear(hidden_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(hidden_dim, args.num_heads, batch_first=True)

    def forward(self, feature1, feature2, feature3):
        # self: feature1 = feature2 = feature3
        # cross: feature1 -- Q, feature2 -- K, feature3 -- V
        # feature: B, D, H, W
        pdb.set_trace()
        B, D, H, W = feature1.shape

        feature1 = feature1.reshape(B, D, -1).transpose(-1, -2)     # B, 1024, 256 (B, N, D)
        feature2 = feature2.reshape(B, D, -1).transpose(-1, -2)
        feature3 = feature3.reshape(B, D, -1).transpose(-1, -2)

        Q = self.linearQ(feature1)
        K = self.linearK(feature1)
        V = self.linearV(feature1)

        attn_output, _ = self.attention(Q, K, V)

        return attn_output