
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class sig_NTM(nn.Module):
    def __init__(self, args, num_class):
        super(sig_NTM, self).__init__()
        self.args = args
        self.num_class = num_class
        self.device = args.device

        T = torch.ones(self.num_class, self.num_class)
        self.register_parameter(name='NTM', param=nn.parameter.Parameter(torch.FloatTensor(T)))
        self.NTM
        
        nn.init.kaiming_normal_(self.NTM, mode='fan_out', nonlinearity='relu')

        self.Identity_prior = torch.eye(self.num_class, self.num_class)

        cd_path = args.task + '_' + args.dataset + '_' + args.noise_type
        Class_dist = np.load('../JCAS/ClassDist/ClassDist_' + cd_path + '.npy')
        self.Class_dist = torch.FloatTensor(np.tile(Class_dist, (self.num_class, 1)))

    def forward(self):
        T = torch.sigmoid(self.NTM).to(self.device)
        T = T.mul(self.Class_dist.to(self.device).detach()) + self.Identity_prior.to(self.device).detach()
        T = F.normalize(T, p=1, dim=1)
        return T

class NTM(nn.Module):
    def __init__(self, args, num_class):
        super(NTM, self).__init__()
        self.args = args
        self.num_class = num_class
        self.device = args.device
        self.init = args.JCAS_init

        self.register_parameter(name='w', param=nn.parameter.Parameter(-self.init * torch.ones(self.num_class, self.num_class)))

        self.w.to(self.device)

        co = torch.ones(self.num_class, self.num_class)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(self.device)
        self.identity = torch.eye(self.num_class).to(self.device)

        # Class_dist = np.load('/home/xiaoqiguo2/Class2affinity/ClassDist/ClassDist_asymmetric50.npy')
        # self.Class_dist = torch.FloatTensor(np.tile(Class_dist, (self.num_class, 1)))

    def forward(self):
        sig = torch.sigmoid(self.w).to(self.device)
        T = self.identity.detach() + sig * self.co.detach() 
        T = F.normalize(T, p=1, dim=1)
        return T