# verify whether the label is cprresponding to the input
import numpy as np
import random
import torch
import os
from PIL import Image
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
from tensorboardX import SummaryWriter  
from data.dataloader_optic_preprocess import REFUGE_Dataset, REFUGE2_Dataset, ORIGA_Dataset, G1020_Dataset
import pdb
from torchvision import transforms

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
parser.add_argument('--num_workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--resize_H', type=int, default=512, help='resize')
parser.add_argument('--resize_W', type=int, default=512, help='resize')

# param to be set
parser.add_argument('--dataset', type=str, default='G1020', help='which dataset to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=12, help='Size of train batch')
parser.add_argument('--test_batch_size', type=int, default=12, help='Size of test batch')

# method
parser.add_argument('--bilinear', type=str2bool, default=False, help='bilinear in UNet')

# save path
parser.add_argument('--save_vis_epoch', type=str2bool, default=False, help='save vis per epoch')
parser.add_argument('--out_path', type=str, default='./test_gt/', help='log folder path')
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

    train_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=train_sampler, drop_last=False)
    val_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.test_batch_size, sampler=val_sampler)

elif dataset == 'REFUGE':
    trainset = REFUGE_Dataset(args.data_root, split='train', output_size=(args.resize_H, args.resize_W))
    valset = REFUGE_Dataset(args.data_root, split='val', output_size=(args.resize_H, args.resize_W))

    train_loader = DataLoader(trainset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(valset, num_workers=args.num_workers, batch_size=args.test_batch_size)

elif dataset == 'REFUGE2':
    trainset = REFUGE2_Dataset(args.data_root, split='train', output_size=(args.resize_H, args.resize_W))
    valset = REFUGE2_Dataset(args.data_root, split='val', output_size=(args.resize_H, args.resize_W))

    train_loader = DataLoader(trainset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(valset, num_workers=args.num_workers, batch_size=args.test_batch_size)

elif dataset == 'ORIGA':
    dataset = ORIGA_Dataset(args.data_root, save_split_dir=args.exp_name, output_size=(args.resize_H, args.resize_W))
    train_sampler, val_sampler = split_set(dataset)

    train_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=train_sampler, drop_last=False)
    val_loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.test_batch_size, sampler=val_sampler)

else:
    io.cprint('unknown dataset')


# ==================
# Save target
# ==================

def create_optic_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[1] = [128, 128, 128]
    colormap[2] = [255, 255, 255]

    return colormap


def colorize(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]

    return Image.fromarray(np.uint8(color_mask))


count = 0

batch_idx = 0

# --------------------- #
# save train data
# --------------------- #
new_data = []
new_noisy_label = []
new_clean_label = []
new_data_name = []


colormap = create_optic_colormap()

for data_all in train_loader:
    data, labels, data_name = data_all[0].to(device), data_all[1].long().to(device).squeeze(), data_all[2]

    batch_size, _, img_H, img_W = data.shape

    if len(labels.shape) < 3:
        labels = labels.unsqueeze(0)

    for ii in range(batch_size):

        # save gt vis
        gray = np.uint8(labels[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = args.save_path + '/train_vis/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_gt.png")
        gray = Image.fromarray(gray)
        color.save(color_path)

        # save raw vis
        toPIL = transforms.ToPILImage()
        pic = toPIL(data[ii])
        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
        pic.save(raw_path)

    count += batch_size
    batch_idx += 1


io.cprint("+++++++++++++++++++++++++end of training+++++++++++++++++++++++++")
io.cprint("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
