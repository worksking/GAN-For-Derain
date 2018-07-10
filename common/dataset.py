import os
from os.path import abspath
from os.path import join as join
import torch
from PIL import Image
from glob import glob
import torch.utils.data as data
import torchvision.transforms as transforms

class Derain_Dataset(data.Dataset):
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
        self.data_path = join(root, 'crop_datasets', 'mini_data')
        self.label_path = join(root, 'crop_datasets', 'mini_label')

        for fn in glob(self.data_path + '/*/*.jpg'):
            self.imgs.append(fn)
        #self.imgs = self.imgs[0:119]
        for fn in glob(self.label_path + '/*/*.jpg'):
            self.labels.append(fn)
        #self.labels = self.labels[0:119]

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
