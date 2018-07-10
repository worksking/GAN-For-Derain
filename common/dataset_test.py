import os
from os.path import abspath
from os.path import join as join
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class train_Dataset(data.Dataset):
    '''
    ### Parameters:
        - root:The root path of dataset
        - data_root: The path of file  'rain_filenames.txt'
        - label_root:The path of file 'origin_filenames.txt'
        - loader:Load a image from a given path
        - transform:Transfrom a image to specified format
        - t_transform:Transform a label to specified format
    '''

    def __init__(self, root, loader=None, transform=transforms.ToTensor()):
        self.imgs = []
        self.labels = []
        self.loader = loader
        self.transform = transform
        self.data_root = join(root, 'rain_mage', 'outdata')
        self.label_path = join(root, 'rain_mage', 'outlabel')

        for path,_,files in os.walk(self.data_root):
            for file in files:
                self.imgs += [abspath(join(self.data_root,file))]
        self.imgs = self.imgs[:4000000]
            
        for path, _, files in os.walk(self.data_root):
            for file in files:
                self.labels += [abspath(join(self.label_path, file))]
        self.labels = self.labels[:4000000]
    def _loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        if self.loader is None:
            self.loader = self._loader

        imgs = self.imgs[index]
        labels = self.labels[index]
        imgs = self.transform(self.loader(imgs))
        labels = self.transform(self.loader(labels))
        return imgs, labels

    def __len__(self):
        return len(self.imgs)


class val_Dataset(data.Dataset):
    '''
    ### Parameters:
        - root:The root path of dataset
        - data_root: The path of file  'rain_filenames.txt'
        - label_root:The path of file 'origin_filenames.txt'
        - loader:Load a image from a given path
        - transform:Transfrom a image to specified format
        - t_transform:Transform a label to specified format
    '''

    def __init__(self, root, loader=None, transform=transforms.ToTensor()):
        self.imgs = []
        self.labels = []
        self.loader = loader
        self.transform = transform
        self.data_root = join(root, 'rain_mage', 'outdata')
        self.label_path = join(root, 'rain_mage', 'outlabel')

        for path,_,files in os.walk(self.data_root):
            for file in files:
                self.imgs += [abspath(join(self.data_root,file))]
        self.imgs = self.imgs[4000000:]
            
        for path, _, files in os.walk(self.data_root):
            for file in files:
                self.labels += [abspath(join(self.label_path, file))]
        self.labels = self.labels[4000000:]
    def _loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        if self.loader is None:
            self.loader = self._loader

        imgs = self.imgs[index]
        labels = self.labels[index]
        imgs = self.transform(self.loader(imgs))
        labels = self.transform(self.loader(labels))
        return imgs, labels

    def __len__(self):
        return len(self.imgs)
