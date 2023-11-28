# v2: modified for UNet
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
from evaluate_breast import evaluate_save_breast, save_filter_prediction_vis_all
from tensorboardX import SummaryWriter  
from data.dataloader_optic_noisy import Noisy_Dataset
from models.baseline import DeepLabV3Plus_Baseline_model, DeepLabV2_Baseline_model, UNet_Baseline_model
from loss_utils import DiceLoss, NR_DiceLoss, Dice_CELoss, NR_Dice_CELoss, IoULoss, IoU_CELoss, ComboLoss, SCELoss, LovaszSoftmaxLoss, FocalLoss
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
parser.add_argument('--backbone', default='resnet101', type=str)
parser.add_argument('--pretrained_resnet_path', default='./pretrained', type=str, help='load saved resnet') 
parser.add_argument('--num_workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--num_class', type=int, default=3, help='number of classes per dataset')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--resize_H', type=int, default=512, help='resize')
parser.add_argument('--resize_W', type=int, default=512, help='resize')
parser.add_argument('--loss_function', type=str, default='CE', help='loss')     # CE, etc

# loss
parser.add_argument('--SCE_alpha', type=float, default=1.0, help='SCE_alpha')
parser.add_argument('--SCE_beta', type=float, default=5.0, help='SCE_beta')
parser.add_argument('--Dice_CE_alpha', type=float, default=5.0, help='Dice_CE_alpha')
parser.add_argument('--Dice_CE_beta', type=float, default=1.0, help='Dice_CE_beta')
parser.add_argument('--IoU_CE_alpha', type=float, default=5.0, help='IoU_CE_alpha')
parser.add_argument('--IoU_CE_beta', type=float, default=1.0, help='IoU_CE_beta')
parser.add_argument('--NR_Dice_CE_alpha', type=float, default=1.0, help='NR_Dice_CE_alpha')
parser.add_argument('--NR_Dice_CE_beta', type=float, default=5.0, help='NR_Dice_CE_beta')
parser.add_argument('--NR_Dice_CE_gamma', type=float, default=1.5, help='NR_Dice_CE_gamma, between 1 and 2')
parser.add_argument('--NR_Dice_gamma', type=float, default=1.5, help='NR_Dice_gamma, between 1 and 2')
parser.add_argument('--Combo_alpha', type=float, default=0.5, help='Combo_alpha')
parser.add_argument('--Combo_ce_ratio', type=float, default=0.5, help='Combo_ce_ratio')
parser.add_argument('--CSS_factor', type=float, default=0.15, help='CSS_factor')
parser.add_argument('--use_CSS_sample', type=str2bool, default=True, help='False: only LovaszSoftmaxLoss')
parser.add_argument('--Focal_alpha1', type=float, default=0.25, help='Focal_alpha_class1')
parser.add_argument('--Focal_alpha2', type=float, default=0.75, help='Focal_alpha_class2')
parser.add_argument('--Focal_alpha3', type=float, default=0.85, help='Focal_alpha_class3')
parser.add_argument('--Focal_gamma', type=float, default=2.0, help='Focal_gamma')

# param to be set
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--dataset', type=str, default='BUSI', help='which dataset to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=12, help='Size of train batch')
parser.add_argument('--test_batch_size', type=int, default=12, help='Size of test batch')
parser.add_argument('--bilinear', type=str2bool, default=False, help='bilinear in UNet')

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

args.Focal_alpha = [args.Focal_alpha1, args.Focal_alpha2, args.Focal_alpha3]

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
elif args.loss_function == 'Dice':
    criterion_seg = DiceLoss(args).to(device)
elif args.loss_function == 'Dice_CE':
    criterion_seg = Dice_CELoss(args).to(device)
elif args.loss_function == 'NR_Dice':
    criterion_seg = NR_DiceLoss(args).to(device)
elif args.loss_function == 'NR_Dice_CE':
    criterion_seg = NR_Dice_CELoss(args).to(device)
elif args.loss_function == 'IoU':
    criterion_seg = IoULoss(args).to(device)
elif args.loss_function == 'IoU_CE':
    criterion_seg = IoU_CELoss(args).to(device)
elif args.loss_function == 'SCE':
    criterion_seg = SCELoss(args).to(device)
elif args.loss_function == 'Combo':
    criterion_seg = ComboLoss(args).to(device)
elif args.loss_function == 'CSS':
    criterion_seg = LovaszSoftmaxLoss(args).to(device)
elif args.loss_function == 'Focal':
    criterion_seg = FocalLoss(args).to(device)
else:
    io.cprint('unknown loss function')
    
# ==================
# Preparation
# ==================
total_iters = len(train_loader) * args.epochs

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

    print_losses = {'total': 0.0}

    if args.loss_function == 'CE':
        print_losses['ce'] = 0.0
    if args.loss_function == 'Focal':
        print_losses['focal'] = 0.0
    if args.loss_function == 'Dice':
        print_losses['dice'] = 0.0
    if args.loss_function == 'Dice_CE':
        print_losses['ce'] = 0.0
        print_losses['dice'] = 0.0
    if args.loss_function == 'NR_Dice':
        print_losses['NR_dice'] = 0.0
    if args.loss_function == 'NR_Dice_CE':
        print_losses['NR_dice'] = 0.0
        print_losses['ce'] = 0.0
    if args.loss_function == 'IoU':
        print_losses['IoU'] = 0.0
    if args.loss_function == 'IoU_CE':
        print_losses['IoU'] = 0.0
        print_losses['ce'] = 0.0
    if args.loss_function == 'SCE':
        print_losses['ce'] = 0.0
        print_losses['rce'] = 0.0
    if args.loss_function == 'Combo':
        print_losses['weighted_ce'] = 0.0
        print_losses['dice'] = 0.0
    if args.loss_function == 'CSS':
        print_losses['LovaszSoftmax'] = 0.0
        print_losses['fg'] = 0.0
        print_losses['background'] = 0.0
    count = 0.0

    batch_idx = 0
    for data_all in train_loader:

        optimizer.zero_grad()

        loss = 0.0

        img, noisy_mask, gt_mask = data_all[0].to(device), data_all[1].long().to(device), data_all[2].long().to(device)
        img_name = data_all[-1]

        batch_size, _, img_H, img_W = img.shape

        # start training process

        logits = model(img)
        preds = logits["pred"].softmax(dim=1).max(dim=1)[1]

        # ============== #
        # calculate loss #
        # ============== #
        
        # seg loss
        if args.loss_function == 'CE':
            loss_seg = criterion_seg(logits["pred"], noisy_mask)

            print_losses['total'] += loss_seg.item() * batch_size

        elif args.loss_function == 'Focal':
            loss_seg = criterion_seg(logits["pred"], noisy_mask)

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['focal'] += loss_seg.item() * batch_size

        elif args.loss_function == 'Dice':
            loss_logits = criterion_seg(logits["pred"], noisy_mask)
            loss_seg = loss_logits['total']
            loss_dice = loss_logits['dice']

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['dice'] += loss_dice.item() * batch_size

        elif args.loss_function == 'Dice_CE':
            loss_logits = criterion_seg(logits["pred"], noisy_mask)
            loss_seg = loss_logits['total']
            loss_dice = loss_logits['dice']
            loss_ce = loss_logits['ce']

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['dice'] += loss_dice.item() * batch_size
            print_losses['ce'] += loss_ce.item() * batch_size

        elif args.loss_function == 'NR_Dice':
            loss_logits = criterion_seg(logits["pred"], noisy_mask)
            loss_seg = loss_logits['total']
            loss_NR_dice = loss_logits['NR_dice']

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['NR_dice'] += loss_NR_dice.item() * batch_size

        elif args.loss_function == 'NR_Dice_CE':
            loss_logits = criterion_seg(logits["pred"], noisy_mask)
            loss_seg = loss_logits['total']
            loss_NR_dice = loss_logits['NR_dice']
            loss_ce = loss_logits['ce']

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['NR_dice'] += loss_NR_dice.item() * batch_size
            print_losses['ce'] += loss_ce.item() * batch_size

        elif args.loss_function == 'IoU':
            loss_logits = criterion_seg(logits["pred"], noisy_mask)
            loss_seg = loss_logits['total']
            loss_IoU = loss_logits['IoU']

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['IoU'] += loss_IoU.item() * batch_size

        elif args.loss_function == 'IoU_CE':
            loss_logits = criterion_seg(logits["pred"], noisy_mask)
            loss_seg = loss_logits['total']
            loss_IoU = loss_logits['IoU']
            loss_ce = loss_logits['ce']

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['IoU'] += loss_IoU.item() * batch_size
            print_losses['ce'] += loss_ce.item() * batch_size

        elif args.loss_function == 'SCE':
            loss_logits = criterion_seg(logits["pred"], noisy_mask)
            loss_seg = loss_logits['total']
            loss_ce = loss_logits['ce']
            loss_rce = loss_logits['rce']

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['rce'] += loss_rce.item() * batch_size
            print_losses['ce'] += loss_ce.item() * batch_size

        elif args.loss_function == 'Combo':
            loss_logits = criterion_seg(logits["pred"], noisy_mask)
            loss_seg = loss_logits['total']
            loss_weighted_ce = loss_logits['weighted_ce']
            loss_dice = loss_logits['dice']

            print_losses['total'] += loss_seg.item() * batch_size
            print_losses['dice'] += loss_dice.item() * batch_size
            print_losses['weighted_ce'] += loss_weighted_ce.item() * batch_size
        
        elif args.loss_function == 'CSS':
            # Effective semantic segmentation in Cataract Surgery: What matters most?
            loss_cat_wise = criterion_seg(logits["pred"], noisy_mask)

            loss_bg = loss_cat_wise[0]
            loss_fg = loss_cat_wise[1]

            if args.use_CSS_sample:
                # account wise occurance
                total_fg = (noisy_mask == 1).sum()
                total_bg = (noisy_mask == 0).sum()

                all_pixel = noisy_mask.shape[-1] * noisy_mask.shape[-2]

                factor_fg = torch.sqrt(args.CSS_factor / (total_fg / all_pixel))

                loss_fg = loss_fg * factor_fg

                loss_seg = (loss_bg + loss_fg) / 2

                print_losses['total'] += loss_seg.item() * batch_size
                print_losses['LovaszSoftmax'] += loss_seg.item() * batch_size
                print_losses['fg'] += loss_fg.item() * batch_size
                print_losses['background'] += loss_bg.item() * batch_size
            else:
                loss_seg = (loss_bg + loss_fg) / 3

                print_losses['total'] += loss_seg.item() * batch_size
                print_losses['LovaszSoftmax'] += loss_seg.item() * batch_size
                print_losses['fg'] += loss_fg.item() * batch_size
                print_losses['background'] += loss_bg.item() * batch_size

        loss = loss_seg

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
            if args.loss_function == 'CE':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)

            elif args.loss_function == 'Focal':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_focal', print_losses['focal'] / (batch_idx+1), iters)

            elif args.loss_function == 'Dice':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_Dice', print_losses['dice'] / (batch_idx+1), iters)

            elif args.loss_function == 'Dice_CE':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_Dice', print_losses['dice'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_CE', print_losses['ce'] / (batch_idx+1), iters)

            elif args.loss_function == 'NR_Dice':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_NR_Dice', print_losses['NR_dice'] / (batch_idx+1), iters)

            elif args.loss_function == 'NR_Dice_CE':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_NR_Dice', print_losses['NR_dice'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_CE', print_losses['ce'] / (batch_idx+1), iters)

            elif args.loss_function == 'IoU':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_IoU', print_losses['IoU'] / (batch_idx+1), iters)

            elif args.loss_function == 'IoU_CE':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_IoU', print_losses['IoU'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_CE', print_losses['ce'] / (batch_idx+1), iters)

            elif args.loss_function == 'SCE':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_CE', print_losses['ce'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_RCE', print_losses['rce'] / (batch_idx+1), iters)

            elif args.loss_function == 'Combo':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_weighted_CE', print_losses['weighted_ce'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_dice', print_losses['dice'] / (batch_idx+1), iters)

            elif args.loss_function == 'CSS':
                tb.add_scalar('train_loss_total', print_losses['total'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_LovaszSoftmax', print_losses['LovaszSoftmax'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_fg', print_losses['fg'] / (batch_idx+1), iters)
                tb.add_scalar('train_loss_background', print_losses['background'] / (batch_idx+1), iters)

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
# test_loss_OD, test_result_OD = evaluate_save_optic('Val', io, device, args.save_path, best_OD_model, val_loader, epoch)
# test_loss_OC, test_result_OC = evaluate_save_optic('Val', io, device, args.save_path, best_OC_model, val_loader, epoch)
io.cprint("+++++++++++++++++++++++++end of training+++++++++++++++++++++++++")
io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
