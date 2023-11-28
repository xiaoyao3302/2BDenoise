# 2B network, compare three fusion modules, i.e., add, add_mapping and cat
# v2: modified for UNet, add Focal Loss
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from evaluate_optic import evaluate_save_optic_2B, save_filter_prediction_vis
from tensorboardX import SummaryWriter  
from data.dataloader_optic_noisy_test import Noisy_Dataset
from models.denoise_2B_fusion import DeepLabV3Plus_2B_Denoise_model, DeepLabV2_2B_Denoise_model, UNet_2B_Denoise_model
from train_utils import filtering_mask_selection_v2, weighting_noisy_mask
from loss_utils import FocalLoss
import pdb

MAX_LOSS = 9 * (10 ** 9)


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA noise')
parser.add_argument('--data_root', default='./save_new_data/', type=str)
parser.add_argument('--model', default='DeepLabv3p', type=str)
parser.add_argument('--bilinear', type=str2bool, default=False, help='bilinear in UNet')
parser.add_argument('--backbone', default='resnet101', type=str)
parser.add_argument('--pretrained_resnet_path', default='./pretrained', type=str, help='load saved resnet') 
parser.add_argument('--num_workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--num_class', type=int, default=3, help='number of classes per dataset')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--resize_H', type=int, default=512, help='resize')
parser.add_argument('--resize_W', type=int, default=512, help='resize')
parser.add_argument('--seg_loss_function', type=str, default='CE', help='loss')              # CE
parser.add_argument('--noisy_seg_loss_function', type=str, default='Focal', help='loss')     # CE or Focal
parser.add_argument('--Focal_alpha', type=float, default=[1, 3, 5], help='Focal_alpha')
parser.add_argument('--Focal_gamma', type=float, default=2.0, help='Focal_gamma')

# param to be set
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--dataset', type=str, default='G1020', help='which dataset to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=12, help='Size of train batch')
parser.add_argument('--test_batch_size', type=int, default=12, help='Size of test batch')

# method
parser.add_argument('--num_warm_up', type=int, default=0, help='0 for no warm up')
parser.add_argument('--use_filtering', type=str2bool, default=True, help='filter out noisy pixels')
parser.add_argument('--use_balance_filtering', type=str2bool, default=True, help='decrease the number of the remained background pixels')
parser.add_argument('--balance_times', type=int, default=5, help='number of remained background pixels should not exceed k times of the foreground pixels')
parser.add_argument('--base_threshold', type=float, default=0.9, help='base threshold for noisy label filtering')
parser.add_argument('--filtering_case', type=str, default='case2', help='clean prediction filtering case')
parser.add_argument('--mode_balance', type=str, default='random', help='random or cluster or random cluster or FPS_cluster, sample background pixels')
parser.add_argument('--num_balance_cluster', type=int, default=5, help='mode_balance=XXXcluster, set the number of the clusters')

# 2B method
parser.add_argument('--fusion_mode', type=str, default='cat', help='cat or add_mapping_noisy or add_mapping_clean or add')
parser.add_argument('--use_detach', type=str2bool, default=False, help='detach the gradient when calculating feature (all)')
parser.add_argument('--use_iterative_train', type=str2bool, default=False, help='bi-directional sum when calculating feature (all)')

# weights
parser.add_argument('--w_clean_seg', type=float, default=1.0, help='weight of clean seg loss')
parser.add_argument('--w_all_seg', type=float, default=1.0, help='weight of all seg loss')

# optimizer
parser.add_argument('--base_lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_backbone', type=float, default=1, help='backbone, 1x finetune')
parser.add_argument('--lr_network', type=float, default=10, help='network, 10x finetune')
parser.add_argument('--poly_power', type=float, default=0.9, help='poly scheduler')

# save path
parser.add_argument('--save_vis_epoch', type=str2bool, default=False, help='save vis per epoch')
parser.add_argument('--out_path', type=str, default='./save_target/', help='log folder path')
parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment')

args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

args.save_path = io.path
tb_dir = args.save_path
tb = SummaryWriter(log_dir=tb_dir)

random.seed(1)
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

args.device = device

# ==================
# Read Data
# ==================

trainset = Noisy_Dataset(args.data_root, args.dataset, split='train', output_size=(args.resize_H, args.resize_W))
valset = Noisy_Dataset(args.data_root, args.dataset, split='val', output_size=(args.resize_H, args.resize_W))

train_loader = DataLoader(trainset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(valset, num_workers=args.num_workers, batch_size=args.test_batch_size)


# ==================
# Init Model
# ==================
if args.model == 'DeepLabv3p':
    model = DeepLabV3Plus_2B_Denoise_model(args)
elif args.model == 'DeepLabv2':
    model = DeepLabV2_2B_Denoise_model(args)
elif args.model == 'UNet':
    model = UNet_2B_Denoise_model(args)
else:
    io.cprint('unknown model')

model = model.to(device)

# first, load saved trained model
io.cprint("-----------------------------------------------------------------")
try:
    checkpoint = torch.load(args.save_path + '/mean_' + args.model + '.pt')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    io.cprint('load saved model')
except:
    start_epoch = 0
    io.cprint('no saved model')

io.cprint('start epoch: %d, total epoch: %d' % (start_epoch, args.epochs))

best_mean_model = copy.deepcopy(model)

# ==================
# Optimizer
# ==================
if args.model == 'UNet':
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr * args.lr_network, momentum=0.9, weight_decay=1e-4)
else:
    optimizer = optim.SGD([{'params': model.clean_encoder.backbone.parameters(), 'lr': args.base_lr * args.lr_backbone},
                        {'params': model.noisy_encoder.backbone.parameters(), 'lr': args.base_lr * args.lr_backbone},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': args.base_lr * args.lr_network}], 
                        lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

# ==================
# Loss
# ==================
if args.seg_loss_function == 'CE':
    criterion_seg = torch.nn.CrossEntropyLoss(reduction='none').to(device)
elif args.seg_loss_function == 'MSE':
    criterion_seg = torch.nn.MSELoss(reduction='none').to(device)
elif args.seg_loss_function == 'nll':
    criterion_seg = torch.nn.NLLLoss(reduction='none').to(device)
else:
    io.cprint('unknown seg loss function')

if args.noisy_seg_loss_function == 'CE':
    criterion_noisy_seg = torch.nn.CrossEntropyLoss(reduction='none').to(device)
elif args.noisy_seg_loss_function == 'Focal':
    criterion_noisy_seg = FocalLoss(args).to(device)
else:
    io.cprint('unknown noisy seg loss function')

# ==================
# Preparation
# ==================
total_iters = len(train_loader) * args.epochs

best_IoU_OD = 0.0
best_Dice_OD = 0.0
best_epoch_OD = 0

best_IoU_OC = 0.0
best_Dice_OC = 0.0
best_epoch_OC = 0

best_IoU_mean = 0.0
best_Dice_mean = 0.0
best_epoch_mean = 0

threshold = args.base_threshold

io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# ===================
# Final Val
# ===================
io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
Val_loss_mean, Val_result_mean = evaluate_save_optic_2B('Val', io, device, args.save_path, best_mean_model, val_loader, idx_epoch=best_epoch_mean, need_save=True)
# test_loss_OD, test_result_OD = evaluate_save_optic('Val', io, device, args.save_path, best_OD_model, val_loader, epoch)
# test_loss_OC, test_result_OC = evaluate_save_optic('Val', io, device, args.save_path, best_OC_model, val_loader, epoch)
io.cprint("+++++++++++++++++++++++++end of training+++++++++++++++++++++++++")
io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
