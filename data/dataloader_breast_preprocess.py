# for constructing noisy data
# breast dataloader
# BUSI: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

# note: root_dor does not contain name of the dataset

# only keeps train and val for all sets, former 80% as train, last 20% as val
# normalized
# replace Image with cv2
# use dataloader from https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch

import os
import numpy as np
import datetime
import torch
from PIL import Image
import cv2
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pdb


def normalize(pixels):
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / (std + 1e-6)
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0   

    return pixels


class BUSI_Dataset(Dataset):
    def __init__(self, root_dir, save_split_dir=None, use_aug=False, output_size=(256,256)):
        self.output_size = output_size
        self.root_dir = os.path.join(root_dir, "BUSI/Dataset_BUSI_with_GT/all")

        self.images = []
        self.segs = []

        # Load data index
        self.image_filenames = []

        for path in os.listdir(self.root_dir):
            if(not path.startswith('.')):
                if(path.split('.')[0].endswith(')')):
                    self.image_filenames.append(path)

        for path in os.listdir(self.root_dir):
            if(not path.startswith('.')):
                if (path.split('.')[0].endswith('1')):
                    # remove the multi task situation (just several)
                    self.image_filenames.remove(path.replace("_mask_1.png", ".png"))
        
        # split train or test
        self.num_examples = len(self.image_filenames)
        self.ind = np.arange(self.num_examples)
        # np.random.shuffle(self.ind)

        num_train = int(self.num_examples * 0.8)

        self.train_ind = self.ind[:num_train]
        idx_train = np.arange(len(self.train_ind))
        np.random.shuffle(idx_train)
        self.train_ind = self.train_ind[idx_train]

        self.val_ind = self.ind[num_train:]
        idx_val = np.arange(len(self.val_ind))
        np.random.shuffle(idx_val)
        self.val_ind = self.val_ind[idx_val]

        # save split
        split_dir = './breast_split/BUSI'
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

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
        for idx in range(len(self.train_ind)):
            f.write(str(self.train_ind[idx]))
            f.write('\n')

        f = open(split_val_name, 'w')
        for idx in range(len(self.val_ind)):
            f.write(str(self.val_ind[idx]))
            f.write('\n')

        # train
        for k in range(len(self.train_ind)):
            print('Loading train image {}/{}...'.format(k, len(self.train_ind)), end='\r')
            # image
            img_name = os.path.join(self.root_dir, self.image_filenames[self.train_ind[k]])
            img = cv2.imread(img_name).astype(np.float32) # [H, W, 3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)[:, :, :3]

            # mask
            seg_name = os.path.join(self.root_dir, self.image_filenames[self.train_ind[k]].replace(".png", "_mask.png"))
            mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask = np.expand_dims(mask, -1) # [H, W, 1] 

            # augmentation
            if use_aug:
                transformations = A.Compose([
                        A.Resize(output_size[0], output_size[1]),
                        A.OneOf([
                            A.RandomRotate90(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                            A.RandomBrightnessContrast(p=0.2),
                            A.GridDistortion(p=0.2),
                            A.ElasticTransform(p=0.2)
                        ]), 
                ])
            else:
                transformations = A.Compose([
                        A.Resize(output_size[0], output_size[1]),
                ])

            transformed = transformations(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

            # img = normalize(img) 
            mask = mask / 255.0 

            img = np.transpose(img, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))

            img = torch.Tensor(img)
            mask = torch.Tensor(mask)

            self.images.append(img)
            self.segs.append(mask)

        # valid
        for k in range(len(self.val_ind)):
            print('Loading val image {}/{}...'.format(k, len(self.val_ind)), end='\r')

            # image
            img_name = os.path.join(self.root_dir, self.image_filenames[self.val_ind[k]])
            img = cv2.imread(img_name).astype(np.float32) # [H, W, 3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)[:, :, :3]

            # mask
            seg_name = os.path.join(self.root_dir, self.image_filenames[self.val_ind[k]].replace(".png", "_mask.png"))
            mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask = np.expand_dims(mask, -1) # [H, W, 1] 

            transformations = A.Compose([
                    A.Resize(output_size[0], output_size[1]),
            ])

            transformed = transformations(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

            # img = normalize(img) 
            mask = mask / 255.0 

            img = np.transpose(img, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))

            img = torch.Tensor(img)
            mask = torch.Tensor(mask)

            self.images.append(img)
            self.segs.append(mask)

            self.images.append(img)
            self.segs.append(mask)

        print('Succesfully loaded BUSI train dataset.' + ' '*50)
        print('number of train samples in BUSI dataset: {}.'.format(len(self.train_ind)) + ' '*50)
        print('Succesfully loaded BUSI val dataset.' + ' '*50)
        print('number of val samples in BUSI dataset: {}.'.format(len(self.val_ind)) + ' '*50)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.image_filenames[idx]
        img = self.images[idx]          # 3, H, W
        # mask = 0, 1, 2
        # mask = 0: background
        # mask = 1: optic disk
        # mask = 2: optic cup
        mask = self.segs[idx]           # 1, H, W

        return img, mask, name


class BUS_Dataset(Dataset):
    def __init__(self, root_dir, save_split_dir=None, use_aug=False, output_size=(256,256)):
        self.output_size = output_size
        self.root_dir = os.path.join(root_dir, "BUS/")
        self.img_dir = os.path.join(self.root_dir, "original/")
        self.mask_dir = os.path.join(self.root_dir, "GT/")

        self.images = []
        self.segs = []

        # Load data index
        self.image_filenames = []

        for path in os.listdir(self.img_dir):
            self.image_filenames.append(path)
        
        # split train or test
        self.num_examples = len(self.image_filenames)
        self.ind = np.arange(self.num_examples)
        # np.random.shuffle(self.ind)

        num_train = int(self.num_examples * 0.8)

        self.train_ind = self.ind[:num_train]
        idx_train = np.arange(len(self.train_ind))
        np.random.shuffle(idx_train)
        self.train_ind = self.train_ind[idx_train]

        self.val_ind = self.ind[num_train:]
        idx_val = np.arange(len(self.val_ind))
        np.random.shuffle(idx_val)
        self.val_ind = self.val_ind[idx_val]

        # save split
        split_dir = './breast_split/BUS'
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

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
        for idx in range(len(self.train_ind)):
            f.write(str(self.train_ind[idx]))
            f.write('\n')

        f = open(split_val_name, 'w')
        for idx in range(len(self.val_ind)):
            f.write(str(self.val_ind[idx]))
            f.write('\n')

        # train
        for k in range(len(self.train_ind)):
            print('Loading train image {}/{}...'.format(k, len(self.train_ind)), end='\r')
            # image
            img_name = os.path.join(self.img_dir, self.image_filenames[self.train_ind[k]])
            img = cv2.imread(img_name).astype(np.float32) # [H, W, 3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)[:, :, :3]

            # mask
            seg_name = os.path.join(self.mask_dir, self.image_filenames[self.train_ind[k]])
            mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask = np.expand_dims(mask, -1) # [H, W, 1] 

            # augmentation
            if use_aug:
                transformations = A.Compose([
                        A.Resize(output_size[0], output_size[1]),
                        A.OneOf([
                            A.RandomRotate90(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                            A.RandomBrightnessContrast(p=0.2),
                            A.GridDistortion(p=0.2),
                            A.ElasticTransform(p=0.2)
                        ]), 
                ])
            else:
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

            img = torch.Tensor(img)
            mask = torch.Tensor(mask)

            self.images.append(img)
            self.segs.append(mask)

        # valid
        for k in range(len(self.val_ind)):
            print('Loading val image {}/{}...'.format(k, len(self.val_ind)), end='\r')

            # image
            img_name = os.path.join(self.img_dir, self.image_filenames[self.val_ind[k]])
            img = cv2.imread(img_name).astype(np.float32) # [H, W, 3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)[:, :, :3]

            # mask
            seg_name = os.path.join(self.mask_dir, self.image_filenames[self.val_ind[k]])
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

            img = torch.Tensor(img)
            mask = torch.Tensor(mask)

            self.images.append(img)
            self.segs.append(mask)

            self.images.append(img)
            self.segs.append(mask)

        print('Succesfully loaded BUS train dataset.' + ' '*50)
        print('number of train samples in BUS dataset: {}.'.format(len(self.train_ind)) + ' '*50)
        print('Succesfully loaded BUS val dataset.' + ' '*50)
        print('number of val samples in BUS dataset: {}.'.format(len(self.val_ind)) + ' '*50)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.image_filenames[idx]
        img = self.images[idx]          # 3, H, W
        # mask = 0, 1, 2
        # mask = 0: background
        # mask = 1: optic disk
        # mask = 2: optic cup
        mask = self.segs[idx]           # 1, H, W

        return img, mask, name
    
if __name__ == '__main__':
    root = '../../data/breast/'
    dataset = BUS_Dataset(root)
    print(dataset[1])
