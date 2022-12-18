import csv
import pathlib
from glob import glob

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data.dataset import Dataset

class_to_idx = {
    'asparagus': 0,
    'bambooshoots': 1,
    'betel': 2,
    'broccoli': 3,
    'cauliflower': 4,
    'chinesecabbage': 5,
    'chinesechives': 6,
    'custardapple': 7,
    'grape': 8,
    'greenhouse': 9,
    'greenonion': 10,
    'kale': 11,
    'lemon': 12,
    'lettuce': 13,
    'litchi': 14,
    'longan': 15,
    'loofah': 16,
    'mango': 17,
    'onion': 18,
    'others': 19,
    'papaya': 20,
    'passionfruit': 21,
    'pear': 22,
    'pennisetum': 23,
    'redbeans': 24,
    'roseapple': 25,
    'sesbania': 26,
    'soybeans': 27,
    'sunhemp': 28,
    'sweetpotato': 29,
    'taro': 30,
    'tea': 31,
    'waterbamboo': 32
}

location = {}
with open('../datasets/CROPS/loc_cnt.txt', 'r') as fr:
    for lines in fr.readlines():
        loc_name, loc_cnt = lines.strip().split(':')
        loc_name = loc_name[1:-1]
        loc_cnt = [int(x) for x in loc_cnt[1:-2].split(',')]
        location[loc_name] = loc_cnt

class TrainDataset(Dataset):
    def __init__(self, root, folds, transform):
        self.transform = transform
        self.class_to_idx = class_to_idx

        samples = []
        for fold in folds:
            samples.extend(glob(f'{root}/{fold}/*/*.jpg'))

        self.samples = sorted(samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        path = pathlib.Path(sample)
        image = Image.open(path).convert('RGB')

        label = class_to_idx[sample.split('/')[-2]]

        if self.transform is not None:
            image = self.transform(image=np.array(image))['image']

        return image, label

    def __len__(self):
        return len(self.samples)

class ValDataset(Dataset):
    def __init__(self, root, folds, size):
        self.transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(),
            ToTensorV2()
        ])

        self.class_to_idx = class_to_idx

        samples = []
        for fold in folds:
            samples.extend(glob(f'{root}/{fold}/*/*.jpg'))

        self.samples = sorted(samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        path = pathlib.Path(sample)
        image = Image.open(path).convert('RGB')

        label = class_to_idx[sample.split('/')[-2]]

        if self.transform is not None:
            image = self.transform(image=np.array(image))['image']

        return image, label

    def __len__(self):
        return len(self.samples)

class TestDataset(Dataset):
    def __init__(self, root, csv_file, size):
        self.transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(),
            ToTensorV2()
        ])

        self.root = root
        self.class_to_idx = class_to_idx
        self.loc_to_prob = location

        file_list = list(csv.DictReader(open(csv_file, 'r', encoding='big5')))
        self.samples = file_list

    def __getitem__(self, index):

        sample = self.samples[index]
        path = pathlib.Path(self.root + sample['Img'])

        image = Image.open(path).convert('RGB')
        town = sample['COUNTYNAME'] + sample['TOWNNAME']
        
        if town in location.keys():
            loc = torch.from_numpy(np.array(location[town], dtype=np.float32))
        else:
            loc = torch.from_numpy(np.array([0]*33, dtype=np.float32))

        if self.transform is not None:
            image = self.transform(image=np.array(image))['image']

        return image, loc, sample['Img']

    def __len__(self):
        return len(self.samples)
