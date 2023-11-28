import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class sig_NTM(nn.Module):
    def __init__(self, args):
        super(sig_NTM, self).__init__()
        self.args = args
        self.num_class = args.num_class

        T = torch.ones(self.num_class, self.num_class)
        self.register_parameter(name='NTM', param=nn.parameter.Parameter(torch.FloatTensor(T)))
        self.NTM
        
        nn.init.kaiming_normal_(self.NTM, mode='fan_out', nonlinearity='relu')

        self.Identity_prior = torch.cat([torch.eye(self.num_class, self.num_class), torch.zeros(self.num_class, self.num_class)], 0)
        cd_path = args.task + '_' + args.dataset
        Class_dist = np.load('../simT/ClassDist/ClassDist_' + cd_path + '.npy')
        # Class_dist = Class_dist / Class_dist.max()
        self.Class_dist = torch.FloatTensor(np.tile(Class_dist, (self.num_class, 1)))

    def forward(self):
        T = torch.sigmoid(self.NTM).cuda()
        T = T.mul(self.Class_dist.cuda().detach()) + self.Identity_prior.cuda().detach()
        T = F.normalize(T, p=1, dim=1)
        return T

class sig_W(nn.Module):
    def __init__(self, args):
        super(sig_W, self).__init__()
        self.args = args
        self.num_class = args.num_class

        init = 1./(self.num_class-1.)

        self.register_parameter(name='weight', param=nn.parameter.Parameter(init*torch.ones(self.num_class, self.num_class)))
        self.weight

        self.identity = torch.zeros(self.num_class, self.num_class) - torch.eye(self.num_class)

    def forward(self):
        ind = np.diag_indices(self.num_class)
        with torch.no_grad():
            self.weight[ind[0], ind[1]] = -10000. * torch.ones(self.num_class).detach()

        w = torch.softmax(self.weight, dim = 1).cuda()

        weight = self.identity.detach().cuda() + w
        return weight
