# 2B network, fusion
# clean fuse with noisy or noisy fuse with clean
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
from evaluate_breast import evaluate_save_breast, save_filter_prediction_vis_all
from tensorboardX import SummaryWriter  
from data.dataloader_optic_noisy import Noisy_Dataset
from models.denoise_2B_fusion import DeepLabV3Plus_2B_Denoise_model, DeepLabV2_2B_Denoise_model, UNet_2B_Denoise_model
from train_utils import filtering_mask_selection_general, filtering_selection_general, weighting_noisy_mask
from loss_utils import DiceLoss, Dice_CELoss, SCELoss
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
parser.add_argument('--data_root', default='./save_new_data/BUSI/DeepLabv3p/test_JED_breast/', type=str)
parser.add_argument('--model', default='DeepLabv3p', type=str)
parser.add_argument('--bilinear', type=str2bool, default=False, help='bilinear in UNet')
parser.add_argument('--backbone', default='resnet101', type=str)
parser.add_argument('--pretrained_resnet_path', default='./pretrained', type=str, help='load saved resnet') 
parser.add_argument('--num_workers', type=int, default=12, help='number of workers in dataloader')
parser.add_argument('--num_class', type=int, default=2, help='number of classes per dataset')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--resize_H', type=int, default=512, help='resize')
parser.add_argument('--resize_W', type=int, default=512, help='resize')

# param to be set
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--dataset', type=str, default='BUSI', help='which dataset to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=2, help='Size of train batch')
parser.add_argument('--test_batch_size', type=int, default=2, help='Size of test batch')

# method
parser.add_argument('--num_warm_up', type=int, default=10, help='0 for no warm up')
parser.add_argument('--use_filtering', type=str2bool, default=True, help='filter out noisy pixels')
parser.add_argument('--use_confident_filtering', type=str2bool, default=True, help='only select confident predictions: case 1, case2, case3')
parser.add_argument('--use_balance_filtering', type=str2bool, default=True, help='decrease the number of the large cat pixels')
parser.add_argument('--mode_balance_filtering', type=str, default='pred', help='pred or label, use prediction or noisy labels to balance sampling')
parser.add_argument('--balance_times', type=int, default=5, help='number of remained max cat pixels should not exceed k times of the min cat pixels')
parser.add_argument('--min_max_cat_number', type=int, default=5000, help='if the minimal cat has 0 pixel, set a lower bound to avoid void case')
parser.add_argument('--base_threshold', type=float, default=0.9, help='base threshold for noisy label filtering')
parser.add_argument('--filtering_case', type=str, default='case3', help='clean prediction filtering case')
parser.add_argument('--use_noisy_filtering', type=str2bool, default=True, help='apply class balance filtering on warm up or noisy stream (no confident, just class balance)')

# 2B method
parser.add_argument('--use_fusion', type=str2bool, default=True, help='apply fusion (cat)')
parser.add_argument('--fusion_mode', type=str, default='clean', help='clean: fuse noisy to clean; clean or noisy')
parser.add_argument('--seg_loss_function', type=str, default='CE', help='loss')
parser.add_argument('--noisy_seg_loss_function', type=str, default='SCE', help='loss')
parser.add_argument('--Dice_CE_alpha', type=float, default=0.5, help='Dice_CE_alpha')
parser.add_argument('--Dice_CE_beta', type=float, default=0.5, help='Dice_CE_beta')
parser.add_argument('--SCE_alpha', type=float, default=0.5, help='SCE_alpha')
parser.add_argument('--SCE_beta', type=float, default=0.5, help='SCE_beta')

# weights
parser.add_argument('--w_clean_seg', type=float, default=1.0, help='weight of clean seg loss')
parser.add_argument('--w_noisy_seg', type=float, default=1.0, help='weight of noisy seg loss')

# optimizer
parser.add_argument('--base_lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_backbone', type=float, default=1, help='backbone, 1x finetune')
parser.add_argument('--lr_network', type=float, default=10, help='network, 10x finetune')
parser.add_argument('--poly_power', type=float, default=0.9, help='poly scheduler')

# save path
parser.add_argument('--save_vis_epoch', type=str2bool, default=True, help='save vis per epoch')
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
    criterion_seg_noisy = torch.nn.CrossEntropyLoss(reduction='none').to(device)
elif args.noisy_seg_loss_function == 'Dice':
    criterion_seg_noisy = DiceLoss(args).to(device)
elif args.noisy_seg_loss_function == 'Dice_CE':
    criterion_seg_noisy = Dice_CELoss(args).to(device)
elif args.noisy_seg_loss_function == 'SCE':
    criterion_seg_noisy = SCELoss(args).to(device)
else:
    io.cprint('unknown noisy loss function')

# ==================
# Preparation
# ==================
total_iters = len(train_loader) * args.epochs

best_IoU_mean = 0.0
best_Dice_mean = 0.0
best_epoch_mean = 0

threshold = args.base_threshold

io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

for epoch in range(start_epoch, args.epochs):

    if args.model == 'UNet':
        io.cprint("current network learning rate: %.4f" % (optimizer.param_groups[0]["lr"]))
        io.cprint("-----------------------------------------------------------------")
    else:
        io.cprint("current backbone1 learning rate: %.4f" % (optimizer.param_groups[0]["lr"]))
        io.cprint("current backbone2 learning rate: %.4f" % (optimizer.param_groups[1]["lr"]))
        io.cprint("current segmentation head learning rate: %.4f" % (optimizer.param_groups[-1]["lr"]))
        io.cprint("-----------------------------------------------------------------")

    model.train()

    print_losses = {'total': 0.0, 'clean_seg': 0.0, 'noisy_seg': 0.0}

    count = 0.0

    same_pred_ratio = 0.0

    batch_idx = 0
    for data_all in train_loader:

        optimizer.zero_grad()

        loss = 0.0

        img, noisy_mask, gt_mask = data_all[0].to(device), data_all[1].long().to(device), data_all[2].to(device).long()
        img_name = data_all[-1]

        batch_size, _, img_H, img_W = img.shape

        # start training process

        logits = model(img)

        # ============== #
        # calculate loss #
        # ============== #

        # ------------------------- #
        # clean seg loss
        # ------------------------- #
        # default: 2B network requires filtering

        # clean: use filtered pixels (pseudo labels) as supervision
        clean_preds = logits['pred'].softmax(dim=1).max(dim=1)[1]

        loss_seg_clean_pred = criterion_seg(logits['pred'], noisy_mask)

        if epoch < args.num_warm_up:
            if args.use_noisy_filtering:
                # warm up stage: class balance sampling
                filtered_mask = filtering_selection_general(args, noisy_mask)
                loss_seg_clean_pred = (loss_seg_clean_pred * filtered_mask).sum() / (filtered_mask.sum() + 1e-6)
                
            else:
                # warm up stage: use all pixels
                filtered_mask = torch.ones([batch_size, img_H, img_W]).bool().to(device)
                loss_seg_clean_pred = loss_seg_clean_pred.mean()

        else:
            # class balance and confident prediction sampling
            filtered_mask = filtering_mask_selection_general(args, logits['pred'], noisy_mask, threshold)
            loss_seg_clean_pred = (loss_seg_clean_pred * filtered_mask).sum() / (filtered_mask.sum() + 1e-6)

        # save mask vis
        if args.save_vis_epoch:
            save_filter_prediction_vis_all(filtered_mask, gt_mask, noisy_mask, img, img_name, args.save_path, epoch)
            
        print_losses['clean_seg'] += loss_seg_clean_pred.item() * batch_size

        # ------------------------- #
        # noisy seg loss
        # ------------------------- #
        # TODO: use all pixels to supervise noisy seg, exist class imbalance problem
        noisy_pred = logits['noisy_pred']
        if args.noisy_seg_loss_function == 'CE':
            if args.use_noisy_filtering:
                loss_seg_noisy_pred = criterion_seg_noisy(noisy_pred, noisy_mask)
                filtered_noisy_mask = filtering_selection_general(args, noisy_mask)
                loss_seg_noisy_pred = (loss_seg_noisy_pred * filtered_noisy_mask).sum() / (filtered_noisy_mask.sum() + 1e-6)
            else:
                loss_seg_noisy_pred = criterion_seg_noisy(noisy_pred, noisy_mask).mean()
                filtered_noisy_mask = torch.ones([batch_size, img_H, img_W]).bool().to(device)
                loss_seg_noisy_pred = loss_seg_noisy_pred.mean()

        elif args.noisy_seg_loss_function == 'Dice':
            loss_seg_noisy_pred = criterion_seg_noisy(noisy_pred, noisy_mask)['total']

        elif args.noisy_seg_loss_function == 'Dice_CE':
            loss_seg_noisy_pred = criterion_seg_noisy(noisy_pred, noisy_mask)['total']
        
        elif args.noisy_seg_loss_function == 'SCE':
            loss_seg_noisy_pred = criterion_seg_noisy(noisy_pred, noisy_mask)['total']

        print_losses['noisy_seg'] += loss_seg_noisy_pred.item() * batch_size


        # all losses
        if epoch < args.num_warm_up:
            loss = loss_seg_clean_pred + loss_seg_noisy_pred
        else:
            loss = args.w_clean_seg * loss_seg_clean_pred + args.w_noisy_seg * loss_seg_noisy_pred

        print_losses['total'] += loss.item() * batch_size

        count += batch_size

        # count same prediction
        pred_clean = logits["pred"].softmax(dim=1).max(dim=1)[1]
        pred_noisy = logits["noisy_pred"].softmax(dim=1).max(dim=1)[1]

        same_pred_ratio += ((pred_clean == pred_noisy).sum() / (batch_size * img_H * img_W + 1e-6)).item() * batch_size

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
            optimizer.param_groups[1]["lr"] = backbone_lr
            for ii in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[ii]['lr'] = seg_lr


            if args.use_confident_filtering:
                # ------------------------------------------------------- #
                filtered_ratio = filtered_mask.sum() / (img_H * img_W * batch_size)
                tb.add_scalar('filtered_pixel_vs_all_pixel', filtered_ratio, iters)

                # ------------------------------------------------------- #
                filtered_prediction = clean_preds * filtered_mask
                filtered_foreground_prediction = (filtered_prediction == 1)
                filtered_background_prediction = (filtered_prediction == 0)

                # filtered cat / filtered pixels
                filtered_foreground_ratio = filtered_foreground_prediction.sum() / (filtered_mask.sum() + 1e-6)
                filtered_background_ratio = filtered_background_prediction.sum() / (filtered_mask.sum() + 1e-6)

                tb.add_scalar('filtered_foreground_vs_filtered_all', filtered_foreground_ratio, iters)
                tb.add_scalar('filtered_background_vs_filtered_all', filtered_background_ratio, iters)

                # ------------------------------------------------------- #
                filtered_prediction = clean_preds * filtered_mask
                filtered_correct_prediction = (filtered_prediction == gt_mask)
                filtered_correct_foreground_prediction = (filtered_prediction == 1) * filtered_correct_prediction
                filtered_correct_background_prediction = (filtered_prediction == 0) * filtered_correct_prediction

                # filtered correct cat / filtered pixels
                filtered_correct_foreground_ratio = filtered_correct_foreground_prediction.sum() / (filtered_mask.sum() + 1e-6)
                filtered_correct_background_ratio = filtered_correct_background_prediction.sum() / (filtered_mask.sum() + 1e-6)

                tb.add_scalar('filtered_correct_foreground_vs_filtered_all', filtered_correct_foreground_ratio, iters)
                tb.add_scalar('filtered_correct_background_vs_filtered_all', filtered_correct_background_ratio, iters)

        batch_idx += 1

    # print progress
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    io.print_loss("Train", epoch, print_losses)

    # ===================
    # Val
    # ===================
    io.cprint("-----------------------------------------------------------------")
    val_loss, val_result = evaluate_save_breast('Val', io, device, args.save_path, model, val_loader, epoch, need_save=args.save_vis_epoch)

    val_results_mean_IoU = val_result["mean_IoU"]
    val_results_mean_Dice = val_result["mean_Dice"]

    tb.add_scalar('mean_IoU', val_result['mean_IoU'], epoch)
    tb.add_scalar('mean_Dice', val_result['mean_Dice'], epoch)

    # save model according to best results
    if  val_results_mean_IoU > best_IoU_mean:
        best_IoU_mean = val_results_mean_IoU
        best_Dice_mean = val_results_mean_Dice
        best_epoch_mean = epoch
        best_mean_model = io.save_model(model, epoch, mode='mean')
    
    io.cprint("-----------------------------------------------------------------")
    io.cprint("epoch %d, " % (epoch))
    io.cprint("previous best val mean IoU: %.4f" % (best_IoU_mean))
    io.cprint("previous best val mean Dice: %.4f" % (best_Dice_mean))
    io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# ===================
# Final Val
# ===================
io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
Val_loss_mean, Val_result_mean = evaluate_save_breast('Val', io, device, args.save_path, best_mean_model, val_loader, idx_epoch=best_epoch_mean, need_save=True)
# test_loss_OD, test_result_OD = evaluate_save_optic_2cls('Val', io, device, args.save_path, best_OD_model, val_loader, epoch)
# test_loss_OC, test_result_OC = evaluate_save_optic_2cls('Val', io, device, args.save_path, best_OC_model, val_loader, epoch)
io.cprint("+++++++++++++++++++++++++end of training+++++++++++++++++++++++++")
io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
