# modified for UNet
import numpy as np
import random
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from evaluate_optic import evaluate_save_optic
from tensorboardX import SummaryWriter  
from data.dataloader_optic_preprocess import REFUGE_Dataset, REFUGE2_Dataset, ORIGA_Dataset, G1020_Dataset
from models.baseline import DeepLabV3Plus_Baseline_model, DeepLabV2_Baseline_model, UNet_Baseline_model
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
parser.add_argument('--data_root', default='../data/optic_seg/', type=str)
parser.add_argument('--model', default='DeepLabv3p', type=str)
parser.add_argument('--backbone', default='resnet101', type=str)
parser.add_argument('--pretrained_resnet_path', default='./pretrained', type=str, help='load saved resnet') 
parser.add_argument('--num_workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--num_class', type=int, default=3, help='number of classes per dataset')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--resize_H', type=int, default=512, help='resize')
parser.add_argument('--resize_W', type=int, default=512, help='resize')
parser.add_argument('--loss_function', type=str, default='CE', help='loss')     # CE, MSE, nll

# param to be set
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--dataset', type=str, default='G1020', help='which dataset to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=6, help='Size of train batch')
parser.add_argument('--test_batch_size', type=int, default=6, help='Size of test batch')

# method
parser.add_argument('--bilinear', type=str2bool, default=False, help='bilinear in UNet')

# optimizer
parser.add_argument('--base_lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_backbone', type=float, default=1, help='backbone, 1x finetune')
parser.add_argument('--lr_network', type=float, default=10, help='network, 10x finetune')
parser.add_argument('--poly_power', type=float, default=0.9, help='poly scheduler')

# save path
parser.add_argument('--save_vis_epoch', type=str2bool, default=False, help='save vis per epoch')
parser.add_argument('--out_path', type=str, default='./save_source/', help='log folder path')
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

# ==================
# Read Data
# ==================
def split_set(dataset):
    """
    Input:
        dataset
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

dataset = args.dataset

if dataset == 'G1020':
    dataset = G1020_Dataset(args.data_root, save_split_dir=args.exp_name, output_size=(args.resize_H, args.resize_W))
    train_sampler, val_sampler = split_set(dataset)

    train_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.test_batch_size, sampler=val_sampler)

elif dataset == 'REFUGE':
    trainset = REFUGE_Dataset(args.data_root, split='train', output_size=(args.resize_H, args.resize_W))
    valset = REFUGE_Dataset(args.data_root, split='val', output_size=(args.resize_H, args.resize_W))

    train_loader = DataLoader(trainset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, num_workers=args.num_workers, batch_size=args.test_batch_size)

elif dataset == 'REFUGE2':
    trainset = REFUGE2_Dataset(args.data_root, split='train', output_size=(args.resize_H, args.resize_W))
    valset = REFUGE2_Dataset(args.data_root, split='val', output_size=(args.resize_H, args.resize_W))

    train_loader = DataLoader(trainset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, num_workers=args.num_workers, batch_size=args.test_batch_size)

elif dataset == 'ORIGA':
    dataset = ORIGA_Dataset(args.data_root, save_split_dir=args.exp_name, output_size=(args.resize_H, args.resize_W))
    train_sampler, val_sampler = split_set(dataset)

    train_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.test_batch_size, sampler=val_sampler)

else:
    io.cprint('unknown dataset')

# ==================
# Init Model
# ==================
if args.model == 'DeepLabv3p':
    model = DeepLabV3Plus_Baseline_model(args)
elif args.model == 'DeepLabv2':
    model = DeepLabV2_Baseline_model(args)
elif args.model == 'UNet':
    model = UNet_Baseline_model(args)
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
    optimizer = optim.SGD([{'params': model.branch.encoder.backbone.parameters(), 'lr': args.base_lr * args.lr_backbone},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': args.base_lr * args.lr_network}], 
                        lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

# ==================
# Loss
# ==================
if args.loss_function == 'CE':
    criterion_seg = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
elif args.loss_function == 'MSE':
    criterion_seg = torch.nn.MSELoss(reduction='mean').to(device)
elif args.loss_function == 'nll':
    criterion_seg = torch.nn.NLLLoss(reduction='mean').to(device)
else:
    io.cprint('unknown loss function')
    
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

io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

for epoch in range(start_epoch, args.epochs):

    if args.model == 'UNet':
        io.cprint("current network learning rate: %.4f" % (optimizer.param_groups[0]["lr"]))
        io.cprint("-----------------------------------------------------------------")
    else:
        io.cprint("current backbone learning rate: %.4f" % (optimizer.param_groups[0]["lr"]))
        io.cprint("current segmentation head learning rate: %.4f" % (optimizer.param_groups[-1]["lr"]))
        io.cprint("-----------------------------------------------------------------")

    model.train()

    print_losses = {'total': 0.0, 'seg': 0.0}

    count = 0.0

    batch_idx = 0
    for data_all in train_loader:

        optimizer.zero_grad()

        loss = 0.0

        img, mask = data_all[0].to(device), data_all[1].long().to(device).squeeze()

        batch_size, _, img_H, img_W = img.shape

        # start training process

        logits = model(img)

        # ============== #
        # calculate loss #
        # ============== #
        
        # seg loss
        loss_seg = criterion_seg(logits["pred"], mask)
        print_losses['seg'] += loss_seg.item() * batch_size
        print_losses['total'] += loss_seg.item() * batch_size
        loss += loss_seg

        count += batch_size

        # update
        loss.backward()
        
        optimizer.step()

        iters = epoch * len(train_loader) + batch_idx

        change_lr = args.base_lr * (1 - iters / total_iters) ** args.poly_power
        backbone_lr = change_lr * args.lr_backbone
        seg_lr = change_lr * args.lr_network
            
        if args.model == 'UNet':
            for ii in range(len(optimizer.param_groups)):
                optimizer.param_groups[ii]['lr'] = seg_lr
        else:
            optimizer.param_groups[0]["lr"] = backbone_lr
            for ii in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[ii]['lr'] = seg_lr

        if (batch_idx % (len(train_loader) // 5) == 0):
            tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
            tb.add_scalar('train_loss_seg', print_losses['seg'] / (batch_idx+1), iters)
        
        batch_idx += 1

    # print progress
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    io.print_loss("Train", epoch, print_losses)

    # ===================
    # Val
    # ===================
    io.cprint("-----------------------------------------------------------------")
    val_loss, val_result = evaluate_save_optic('Val', io, device, args.save_path, model, val_loader, epoch, need_save=args.save_vis_epoch)

    val_results_mean_IoU = val_result["mean_IoU"]
    val_results_mean_Dice = val_result["mean_Dice"]
    val_results_OD_IoU = val_result["OD_IoU"]
    val_results_OD_Dice = val_result["OD_Dice"]
    val_results_OC_IoU = val_result["OC_IoU"]
    val_results_OC_Dice = val_result["OC_Dice"]

    tb.add_scalar('mean_IoU', val_result['mean_IoU'], epoch)
    tb.add_scalar('mean_Dice', val_result['mean_Dice'], epoch)
    tb.add_scalar('OD_IoU', val_result['OD_IoU'], epoch)
    tb.add_scalar('OD_Dice', val_result['OD_Dice'], epoch)
    tb.add_scalar('OC_IoU', val_result['OC_IoU'], epoch)
    tb.add_scalar('OC_Dice', val_result['OC_Dice'], epoch)

    # save model according to best results
    if  val_results_mean_IoU > best_IoU_mean:
        best_IoU_mean = val_results_mean_IoU
        best_Dice_mean = val_results_mean_Dice
        best_epoch_mean = epoch
        best_mean_model = io.save_model(model, epoch, mode='mean')

    if  val_results_OD_IoU > best_IoU_OD:
        best_IoU_OD = val_results_OD_IoU
        best_Dice_OD = val_results_OD_Dice
        best_epoch_OD = epoch
        best_OD_model = io.save_model(model, epoch, mode='OD')

    if  val_results_OC_IoU > best_IoU_OC:
        best_IoU_OC = val_results_OC_IoU
        best_Dice_OC = val_results_OC_Dice
        best_epoch_OC = epoch
        best_OC_model = io.save_model(model, epoch, mode='OC')
    
    io.cprint("-----------------------------------------------------------------")
    io.cprint("epoch %d, " % (epoch))
    io.cprint("previous best val mean IoU: %.4f" % (best_IoU_mean))
    io.cprint("previous best val mean Dice: %.4f" % (best_Dice_mean))
    io.cprint("previous best val OD IoU: %.4f" % (best_IoU_OD))
    io.cprint("previous best val OD Dice: %.4f" % (best_Dice_OD))
    io.cprint("previous best val OC IoU: %.4f" % (best_IoU_OC))
    io.cprint("previous best val OC Dice: %.4f" % (best_Dice_OC))
    io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# ===================
# Final Val
# ===================
io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
Val_loss_mean, Val_result_mean = evaluate_save_optic('Val', io, device, args.save_path, best_mean_model, val_loader, idx_epoch=best_epoch_mean, need_save=True)
# test_loss_OD, test_result_OD = evaluate_save_optic('Val', io, device, args.save_path, best_OD_model, val_loader, epoch)
# test_loss_OC, test_result_OC = evaluate_save_optic('Val', io, device, args.save_path, best_OC_model, val_loader, epoch)
io.cprint("+++++++++++++++++++++++++end of training+++++++++++++++++++++++++")
io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
