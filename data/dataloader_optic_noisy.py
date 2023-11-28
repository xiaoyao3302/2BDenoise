# noisy dataloader

import os
import numpy as np
import datetime
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pdb


class Noisy_Dataset(Dataset):
    def __init__(self, root_dir, dataset_name, split='train', output_size=(256,256)):
        self.split = split
        self.output_size = output_size

        root_dir = os.path.join(root_dir, split + '_data')

        if split == 'train':
            self.images = np.load(root_dir + '/new_data.npy')
            self.noisy_labels = np.load(root_dir + '/new_noisy_label.npy')
            self.clean_labels = np.load(root_dir + '/new_clean_label.npy')
            self.images_name = np.load(root_dir + '/new_data_name.npy')

        else:
            self.images = np.load(root_dir + '/new_val_data.npy')
            self.clean_labels = np.load(root_dir + '/new_val_clean_label.npy')
            self.images_name = np.load(root_dir + '/new_val_data_name.npy')
        
        print('Succesfully loaded {} {} dataset.'.format(dataset_name, split) + ' '*50)
        print('number of samples in {} {} dataset: {}.'.format(dataset_name, split, len(self.images)) + ' '*50)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # label = 0, 1, 2
        # label = 0: background
        # label = 1: optic disk
        # label = 2: optic cup

        if self.split == 'train':
            img = self.images[idx]
            noisy_label = self.noisy_labels[idx]
            clean_label = self.clean_labels[idx]
            img_name = self.images_name[idx]

            return img, noisy_label, clean_label, img_name
        else:
            img = self.images[idx]
            clean_label = self.clean_labels[idx]
            img_name = self.images_name[idx]

            return img, clean_label, img_name
    

if __name__ == '__main__':
    root = '../../data/optic_seg/'
    dataset = Noisy_Dataset(root, 'G1020')
    print(dataset[1])
