import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class sig_t(nn.Module):
    def __init__(self, args):
        super(sig_t, self).__init__()
        self.device = args.device
        self.init = args.VolMin_init
        self.num_class = args.num_class
        
        self.register_parameter(name='w', param=nn.parameter.Parameter(-self.init*torch.ones(self.num_class, self.num_class)))

        self.w.to(self.device)

        co = torch.ones(self.num_class, self.num_class)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(self.device)
        self.identity = torch.eye(self.num_class).to(self.device)


    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)

        return T