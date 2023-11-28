# save target data with coarse annotation noise (JED)
# v2: save BUSI or BUI
import argparse
import utils.log
import numpy as np
import random
import os
import datetime
from PIL import Image
import torch
from torchvision import transforms
import cv2
import albumentations as A
import pdb



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
parser.add_argument('--data_root', default='../data/breast/', type=str)
parser.add_argument('--model', default='DeepLabv3p', type=str)
parser.add_argument('--backbone', default='resnet101', type=str)
parser.add_argument('--pretrained_resnet_path', default='./pretrained', type=str, help='load saved resnet') 
parser.add_argument('--num_workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--num_class', type=int, default=2, help='number of classes per dataset')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--resize_H', type=int, default=512, help='resize')
parser.add_argument('--resize_W', type=int, default=512, help='resize')
parser.add_argument('--loss_function', type=str, default='CE', help='loss')     # CE, MSE, nll

# param to be set
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--dataset', type=str, default='BUSI', help='which dataset to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=6, help='Size of train batch')
parser.add_argument('--test_batch_size', type=int, default=6, help='Size of test batch')
parser.add_argument('--use_data_aug', type=str2bool, default=False, help='use data augmentation or not')

# method
parser.add_argument('--bilinear', type=str2bool, default=False, help='bilinear in UNet')
parser.add_argument('--kernel_size', type=int, default=5, help='kernel size of erode noise')
parser.add_argument('--dilate_iteration', type=int, default=8, help='dilate iteration')

# optimizer
parser.add_argument('--base_lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_backbone', type=float, default=1, help='backbone, 1x finetune')
parser.add_argument('--lr_network', type=float, default=10, help='network, 10x finetune')
parser.add_argument('--poly_power', type=float, default=0.9, help='poly scheduler')

# save path
parser.add_argument('--load_path', type=str, default='./save_source/G1020/test', help='load saved source model')
parser.add_argument('--save_vis_epoch', type=str2bool, default=False, help='save vis per epoch')
parser.add_argument('--out_path', type=str, default='./save_new_data/', help='log folder path')
parser.add_argument('--exp_name', type=str, default='test_JED_breast', help='Name of the experiment')

args = parser.parse_args()

output_size = (args.resize_H, args.resize_W)

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

args.save_path = io.path
root_dir = os.path.join(args.data_root, args.dataset)

image_filenames = []

if args.dataset == 'BUSI':
    for path in os.listdir(os.path.join(root_dir, "Dataset_BUSI_with_GT/all")):
        if(not path.startswith('.')):
            if(path.split('.')[0].endswith(')')):
                image_filenames.append(path)

    for path in os.listdir(os.path.join(root_dir, "Dataset_BUSI_with_GT/all")):
        if(not path.startswith('.')):
            if (path.split('.')[0].endswith('1')):
                # remove the multi task situation (just several)
                image_filenames.remove(path.replace("_mask_1.png", ".png"))

elif args.dataset == 'BUS':
    for path in os.listdir(os.path.join(root_dir, "original/")):
        image_filenames.append(path)

else:
    io.cprint('unkwon dataset')

# split train or test
num_examples = len(image_filenames)
ind = np.arange(num_examples)
# np.random.shuffle(self.ind)

num_train = int(num_examples * 0.8)

train_ind = ind[:num_train]
idx_train = np.arange(len(train_ind))
np.random.shuffle(idx_train)
train_ind = train_ind[idx_train]

val_ind = ind[num_train:]
idx_val = np.arange(len(val_ind))
np.random.shuffle(idx_val)
val_ind = val_ind[idx_val]

# save split
if args.dataset == 'BUSI':  
    split_dir = './breast_split/BUSI'
elif args.dataset == 'BUS':
    split_dir = './breast_split/BUS'
else:
    io.cprint('unkwon dataset')

if not os.path.exists(split_dir):
    os.makedirs(split_dir)

save_split_dir = 'save_erode_dilate_noise/'
if save_split_dir is not None:
    save_split_dir = os.path.join(split_dir, save_split_dir)
    if not os.path.exists(save_split_dir):
        os.makedirs(save_split_dir)

    split_train_name = os.path.join(save_split_dir, "train.txt")
    split_val_name = os.path.join(save_split_dir, "val.txt")
else:
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    save_split_dir = os.path.join(split_dir, timestr)
    if not os.path.exists(save_split_dir):
        os.makedirs(save_split_dir)
        
    split_train_name = os.path.join(save_split_dir, "train.txt")
    split_val_name = os.path.join(save_split_dir, "val.txt")

f = open(split_train_name, 'w')
for idx in range(len(train_ind)):
    f.write(str(train_ind[idx]))
    f.write('\n')

f = open(split_val_name, 'w')
for idx in range(len(val_ind)):
    f.write(str(val_ind[idx]))
    f.write('\n')

# --------------------------- #
# train
# --------------------------- #

new_data = []
new_noisy_label = []
new_clean_label = []
new_data_name = []

def normalize(pixels):
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / (std + 1e-6)
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0   

    return pixels

# save new data
if args.dataset == 'BUSI':
    for k in range(len(train_ind)):
        print('Loading train image {}/{}...'.format(k, len(train_ind)), end='\r')
        img_name = os.path.join(root_dir, "Dataset_BUSI_with_GT/all", image_filenames[train_ind[k]])
        img = cv2.imread(img_name).astype(np.float32) # [H, W, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_name = os.path.join(root_dir, "Dataset_BUSI_with_GT/all", image_filenames[train_ind[k]].replace(".png", "_mask.png"))
        mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = np.expand_dims(mask, -1) # [H, W, 1] 

        transformations = A.Compose([
            A.Resize(output_size[0], output_size[1]),
        ])

        transformed = transformations(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        img = normalize(img) 
        mask = mask / 255.0 

        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        img = torch.Tensor(img)                     # 3, H, W
        mask = torch.Tensor(mask)                   # 1, H, W

        new_data.append(img.unsqueeze(0))
        new_data_name.append(image_filenames[train_ind[k]])
        new_clean_label.append(mask)

        # noisy label
        kernel_size = random.randint(2, args.kernel_size)
        dilate_iteration = random.randint(3, args.dilate_iteration)
        # kernel_size = args.kernel_size
        # dilate_iteration = args.dilate_iteration

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        mask_JED = mask.numpy()

        if mask_JED.sum() > 0:
            if random.random() < 0.5:
                mask_JED = cv2.erode(mask_JED, kernel)
            else:
                mask_JED = cv2.dilate(mask_JED, kernel, iterations=dilate_iteration)
        else:
            mask_JED = mask_JED

        mask_JED = torch.Tensor(mask_JED)  
        new_noisy_label.append(mask_JED)

elif args.dataset == 'BUS':
    for k in range(len(train_ind)):
        print('Loading train image {}/{}...'.format(k, len(train_ind)), end='\r')
        img_name = os.path.join(root_dir, "original", image_filenames[train_ind[k]])
        img = cv2.imread(img_name).astype(np.float32) # [H, W, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_name = os.path.join(root_dir, "GT", image_filenames[train_ind[k]])
        mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = np.expand_dims(mask, -1) # [H, W, 1] 

        transformations = A.Compose([
            A.Resize(output_size[0], output_size[1]),
        ])

        transformed = transformations(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        img = normalize(img) 
        mask = mask / 255.0 

        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        img = torch.Tensor(img)                     # 3, H, W
        mask = torch.Tensor(mask)                   # 1, H, W

        new_data.append(img.unsqueeze(0))
        new_data_name.append(image_filenames[train_ind[k]])
        new_clean_label.append(mask)

        # noisy label
        kernel_size = random.randint(2, args.kernel_size)
        dilate_iteration = random.randint(3, args.dilate_iteration)
        # kernel_size = args.kernel_size
        # dilate_iteration = args.dilate_iteration

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        mask_JED = mask.numpy()

        if mask_JED.sum() > 0:
            if random.random() < 0.5:
                mask_JED = cv2.erode(mask_JED, kernel)
            else:
                mask_JED = cv2.dilate(mask_JED, kernel, iterations=dilate_iteration)
        else:
            mask_JED = mask_JED

        mask_JED = torch.Tensor(mask_JED)  
        new_noisy_label.append(mask_JED)

else:
    io.cprint('unknown dataset')

    # toPIL = transforms.ToPILImage()
    # save_data = img
    # pic = toPIL(save_data)
    # color_folder = args.save_path + '/train_vis/'
    # os.makedirs(color_folder, exist_ok=True)
    # image_name = image_filenames[train_ind[k]]
    # raw_path = os.path.join(color_folder, image_name.split('.')[0] + "_raw2.jpg")
    # pic.save(raw_path)

# --------------------------- #
# val
# --------------------------- #

new_val_data = []
new_clean_val_label = []
new_data_val_name = []

if args.dataset == 'BUSI':
    for k in range(len(val_ind)):
        print('Loading val image {}/{}...'.format(k, len(val_ind)), end='\r')

        img_name = os.path.join(root_dir, "Dataset_BUSI_with_GT/all", image_filenames[val_ind[k]])
        img = cv2.imread(img_name).astype(np.float32) # [H, W, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_name = os.path.join(root_dir, "Dataset_BUSI_with_GT/all", image_filenames[val_ind[k]].replace(".png", "_mask.png"))
        mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = np.expand_dims(mask, -1) # [H, W, 1] 

        transformations = A.Compose([
            A.Resize(output_size[0], output_size[1]),
        ])

        transformed = transformations(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        img = normalize(img) 
        mask = mask / 255.0 

        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        img = torch.Tensor(img)             # 3, H, W
        mask = torch.Tensor(mask)           # 1, H, W

        new_val_data.append(img.unsqueeze(0))
        new_data_val_name.append(image_filenames[train_ind[k]])
        new_clean_val_label.append(mask)

elif args.dataset == 'BUS':
    for k in range(len(val_ind)):
        print('Loading val image {}/{}...'.format(k, len(val_ind)), end='\r')

        img_name = os.path.join(root_dir, "original", image_filenames[val_ind[k]])
        img = cv2.imread(img_name).astype(np.float32) # [H, W, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_name = os.path.join(root_dir, "GT", image_filenames[val_ind[k]])
        mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = np.expand_dims(mask, -1) # [H, W, 1] 

        transformations = A.Compose([
            A.Resize(output_size[0], output_size[1]),
        ])

        transformed = transformations(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        img = normalize(img) 
        mask = mask / 255.0 

        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        img = torch.Tensor(img)             # 3, H, W
        mask = torch.Tensor(mask)           # 1, H, W

        new_val_data.append(img.unsqueeze(0))
        new_data_val_name.append(image_filenames[train_ind[k]])
        new_clean_val_label.append(mask)

else:
    io.cprint('unknown dataset')

print('Succesfully loaded train dataset.' + ' '*50)
print('number of train samples in the dataset: {}.'.format(len(train_ind)) + ' '*50)
print('Succesfully loaded val dataset.' + ' '*50)
print('number of val samples in the dataset: {}.'.format(len(val_ind)) + ' '*50)

# pdb.set_trace()

new_data = np.array(torch.cat(new_data, dim=0))
new_noisy_label = np.array(torch.cat(new_noisy_label, dim=0))
new_clean_label = np.array(torch.cat(new_clean_label, dim=0))
new_data_name = np.array(new_data_name)

new_val_data = np.array(torch.cat(new_val_data, dim=0))
new_clean_val_label = np.array(torch.cat(new_clean_val_label, dim=0))
new_data_val_name = np.array(new_data_val_name)

# save vis
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

colormap = create_optic_colormap()

# pdb.set_trace()

for ii in range(len(train_ind)):
    image_name = new_data_name[ii].split('.')[0]  

    # ------------------------------------------------------------------ #
    # save noisy vis
    gray = new_noisy_label[ii]
    gray = np.int8(gray)
    color = colorize(gray, colormap)
    color_folder = args.save_path + '/train_vis/'
    os.makedirs(color_folder, exist_ok=True)
    color_path = os.path.join(color_folder, image_name + "_noisy.png")
    gray = Image.fromarray(gray)
    color.save(color_path)

    # save gt vis
    gray = new_clean_label[ii]
    gray = np.int8(gray)
    color = colorize(gray, colormap)
    color_folder = args.save_path + '/train_vis/'
    os.makedirs(color_folder, exist_ok=True)
    color_path = os.path.join(color_folder, image_name + "_gt.png")
    gray = Image.fromarray(gray)
    color.save(color_path)

    # save raw vis
    raw = new_data[ii]
    toPIL = transforms.ToPILImage()
    save_data = torch.Tensor(raw)
    pic = toPIL(save_data)
    color_folder = args.save_path + '/train_vis/'
    os.makedirs(color_folder, exist_ok=True)
    raw_path = os.path.join(color_folder, image_name.split('.')[0] + "_raw.jpg")
    pic.save(raw_path)

# save data
save_data_path = args.save_path + '/train_data/'
os.makedirs(save_data_path, exist_ok=True)
np.save(save_data_path + 'new_data.npy', new_data)
np.save(save_data_path + 'new_noisy_label.npy', new_noisy_label)
np.save(save_data_path + 'new_clean_label.npy', new_clean_label)
np.save(save_data_path + 'new_data_name.npy', new_data_name)

io.cprint("save noisy train data for %s" % (args.dataset))

save_val_data_path = args.save_path + '/val_data/'
os.makedirs(save_val_data_path, exist_ok=True)
np.save(save_val_data_path + 'new_val_data.npy', new_val_data)
np.save(save_val_data_path + 'new_val_clean_label.npy', new_clean_val_label)
np.save(save_val_data_path + 'new_val_data_name.npy', new_data_val_name)

io.cprint("save noisy val data for %s" % (args.dataset))
