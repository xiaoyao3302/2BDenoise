# for constructing noisy data
# optic dataloader
# REFUGE: including 400 train, 400 val and 400 test (official split, use as target dataset), 2020
# ORIGA: 650 in total, we randomly split 80% train and 20% test, 2010
# G1020: 1200 in total, random 80% train and 20% test, 2020 (preferred as source)
# note that we further divide the train set of ORIGA and G1020 to 60% train and 20% val

# note: root_dor does not contain name of the dataset

# add REFUGE2
# only keeps train and val for all sets, former 80% as train, last 20% as val
# normalized

import os
import numpy as np
import datetime
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pdb


class REFUGE_Dataset(Dataset):
    def __init__(self, root_dir, split='train', output_size=(256,256)):
        self.output_size = output_size
        self.root_dir = os.path.join(root_dir, "REFUGE")
        self.split = split

        self.images = []
        self.segs = []

        # Load data index
        self.image_filenames = []
        self.root_dir = os.path.join(self.root_dir, split)
        for path in os.listdir(os.path.join(self.root_dir, "Images")):
            if(not path.startswith('.')):
                self.image_filenames.append(path)
        
        if split == 'train':
            for k in range(len(self.image_filenames)):
                print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                img_name = os.path.join(self.root_dir, "Images", self.image_filenames[k])
                # img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
                img = Image.open(img_name).convert('RGB')
                img = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])(img)
                img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)
                self.images.append(img)

            for k in range(len(self.image_filenames)):
                print('Loading {} segmentation {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                seg_name = os.path.join(self.root_dir, "Masks", self.image_filenames[k][:-3] + "png")
                mask = np.array(Image.open(seg_name, mode='r'))
                mask = torch.from_numpy(mask[None,:,:])
                mask = transforms.functional.resize(mask, output_size, interpolation=Image.NEAREST)
                self.segs.append(mask)
        else:
            for k in range(len(self.image_filenames)):
                print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                img_name = os.path.join(self.root_dir, "Images", self.image_filenames[k])
                # img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
                img = Image.open(img_name).convert('RGB')
                img = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])(img)
                self.images.append(img)

            for k in range(len(self.image_filenames)):
                print('Loading {} segmentation {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                seg_name = os.path.join(self.root_dir, "Masks", self.image_filenames[k][:-3] + "png")
                mask = np.array(Image.open(seg_name, mode='r'))
                mask = torch.from_numpy(mask[None,:,:])
                self.segs.append(mask)
        print('Succesfully loaded REFUGE {} dataset.'.format(split) + ' '*50)
        print('number of samples in REFUGE {} dataset: {}.'.format(split, len(self.images)) + ' '*50)

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


class REFUGE2_Dataset(Dataset):
    def __init__(self, root_dir, split='train', output_size=(256,256)):
        self.output_size = output_size
        self.root_dir = os.path.join(root_dir, "REFUGE2")
        self.split = split

        self.images = []
        self.segs = []

        # Load data index
        self.image_filenames = []
        self.root_dir = os.path.join(self.root_dir, split)
        for path in os.listdir(os.path.join(self.root_dir, "Images")):
            if(not path.startswith('.')):
                self.image_filenames.append(path)
        
        if split == 'train':
            for k in range(len(self.image_filenames)):
                print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                img_name = os.path.join(self.root_dir, "Images", self.image_filenames[k])
                # img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
                img = Image.open(img_name).convert('RGB')
                img = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])(img)
                img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)
                self.images.append(img)

            for k in range(len(self.image_filenames)):
                print('Loading {} segmentation {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                seg_name = os.path.join(self.root_dir, "Masks", self.image_filenames[k][:-3] + "png")
                mask = np.array(Image.open(seg_name, mode='r'))
                mask = torch.from_numpy(mask[None,:,:])
                mask = transforms.functional.resize(mask, output_size, interpolation=Image.NEAREST)
                self.segs.append(mask)
        else:
            for k in range(len(self.image_filenames)):
                print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                img_name = os.path.join(self.root_dir, "Images", self.image_filenames[k])
                # img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
                img = Image.open(img_name).convert('RGB')
                img = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])(img)
                self.images.append(img)

            for k in range(len(self.image_filenames)):
                print('Loading {} segmentation {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                seg_name = os.path.join(self.root_dir, "Masks", self.image_filenames[k][:-3] + "bmp")
                mask_255 = np.array(Image.open(seg_name, mode='r'))
                mask_255 = torch.from_numpy(mask_255[None,:,:])
                mask = mask_255.clone()
                mask[mask_255 == 0] = 2
                mask[mask_255 == 128] = 1
                mask[mask_255 == 255] = 0
                self.segs.append(mask)
        print('Succesfully loaded REFUGE2 {} dataset.'.format(split) + ' '*50)
        print('number of samples in REFUGE2 {} dataset: {}.'.format(split, len(self.images)) + ' '*50)
            
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
    

class G1020_Dataset(Dataset):
    def __init__(self, root_dir, save_split_dir=None, output_size=(256,256)):
        self.output_size = output_size
        self.root_dir = os.path.join(root_dir, "G1020")

        self.images = []
        self.segs = []

        # Load data index
        self.image_filenames = []
        for path in os.listdir(os.path.join(self.root_dir, "Images_Square")):
            if(not path.startswith('.')):
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
        split_dir = './optic_split/G1020'
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

        for k in range(len(self.train_ind)):
            print('Loading train image {}/{}...'.format(k, len(self.train_ind)), end='\r')
            img_name = os.path.join(self.root_dir, "Images_Square", self.image_filenames[self.train_ind[k]])
            #img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
            img = Image.open(img_name).convert('RGB')
            img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])(img)
            img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)
            self.images.append(img)

        for k in range(len(self.train_ind)):
            print('Loading train segmentation {}/{}...'.format(k, len(self.train_ind)), end='\r')
            seg_name = os.path.join(self.root_dir, "Masks_Square", self.image_filenames[self.train_ind[k]][:-3] + "png")
            mask = np.array(Image.open(seg_name, mode='r'))
            mask = torch.from_numpy(mask[None,:,:])
            mask = transforms.functional.resize(mask, output_size, interpolation=Image.NEAREST)
            self.segs.append(mask)
        
        for k in range(len(self.val_ind)):
            print('Loading val image {}/{}...'.format(k, len(self.val_ind)), end='\r')
            img_name = os.path.join(self.root_dir, "Images_Square", self.image_filenames[self.val_ind[k]])
            #img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
            img = Image.open(img_name).convert('RGB')
            img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])(img)
            self.images.append(img)

        for k in range(len(self.val_ind)):
            print('Loading val segmentation {}/{}...'.format(k, len(self.val_ind)), end='\r')
            seg_name = os.path.join(self.root_dir, "Masks_Square", self.image_filenames[self.val_ind[k]][:-3] + "png")
            mask = np.array(Image.open(seg_name, mode='r'))
            mask = torch.from_numpy(mask[None,:,:])
            self.segs.append(mask)

        print('Succesfully loaded G1020 train dataset.' + ' '*50)
        print('number of train samples in G1020 dataset: {}.'.format(len(self.train_ind)) + ' '*50)
        print('Succesfully loaded G1020 val dataset.' + ' '*50)
        print('number of val samples in G1020 dataset: {}.'.format(len(self.val_ind)) + ' '*50)

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


class ORIGA_Dataset(Dataset):
    def __init__(self, root_dir, save_split_dir=None, output_size=(256,256)):
        self.output_size = output_size
        self.root_dir = os.path.join(root_dir, "ORIGA")

        self.images = []
        self.segs = []

        # Load data index
        self.image_filenames = []
        for path in os.listdir(os.path.join(self.root_dir, "Images_Square")):
            if(not path.startswith('.')):
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
        split_dir = './optic_split/ORIGA'
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

        for k in range(len(self.train_ind)):
            print('Loading train image {}/{}...'.format(k, len(self.train_ind)), end='\r')
            img_name = os.path.join(self.root_dir, "Images_Square", self.image_filenames[self.train_ind[k]])
            #img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
            img = Image.open(img_name).convert('RGB')
            img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])(img)
            img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)
            self.images.append(img)

        for k in range(len(self.train_ind)):
            print('Loading train segmentation {}/{}...'.format(k, len(self.train_ind)), end='\r')
            seg_name = os.path.join(self.root_dir, "Masks_Square", self.image_filenames[self.train_ind[k]][:-3] + "png")
            mask = np.array(Image.open(seg_name, mode='r'))
            mask = torch.from_numpy(mask[None,:,:])
            mask = transforms.functional.resize(mask, output_size, interpolation=Image.NEAREST)
            self.segs.append(mask)
        
        for k in range(len(self.val_ind)):
            print('Loading val image {}/{}...'.format(k, len(self.val_ind)), end='\r')
            img_name = os.path.join(self.root_dir, "Images_Square", self.image_filenames[self.val_ind[k]])
            #img = remove_nerves(np.array(Image.open(img_name).convert('RGB'))).astype(np.float32)
            img = Image.open(img_name).convert('RGB')
            img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])(img)
            self.images.append(img)

        for k in range(len(self.val_ind)):
            print('Loading val segmentation {}/{}...'.format(k, len(self.val_ind)), end='\r')
            seg_name = os.path.join(self.root_dir, "Masks_Square", self.image_filenames[self.val_ind[k]][:-3] + "png")
            mask = np.array(Image.open(seg_name, mode='r'))
            mask = torch.from_numpy(mask[None,:,:])
            self.segs.append(mask)

        print('Succesfully loaded ORIGA train dataset.' + ' '*50)
        print('number of train samples in G1020 dataset: {}.'.format(len(self.train_ind)) + ' '*50)
        print('Succesfully loaded ORIGA val dataset.' + ' '*50)
        print('number of val samples in G1020 dataset: {}.'.format(len(self.val_ind)) + ' '*50)
            
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
    root = '../../data/optic_seg/'
    dataset = REFUGE2_Dataset(root)
    print(dataset[1])
