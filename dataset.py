import os
import glob
from PIL import Image
import torch
import random
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, mode='train'):
        self.transforms = transforms
        self.day = sorted(glob.glob(os.path.join(root, 'day', '*.jpg')))
        self.night = sorted(glob.glob(os.path.join(root, 'night', '*.jpg')))
        assert len(self.day) > 0, "Make sure you downloaded the images!"


    def __getitem__(self, index):
        if index >= len(self.day):
            day_path = random.choice(self.day)
            night_path = self.night[index]
        elif index >= len(self.night):
            day_path = self.day[index]
            night_path = random.choice(self.night)
        else:
            day_path = self.day[index]
            night_path = self.night[index]
            
        day, night = Image.open(day_path).convert('RGB'), Image.open(night_path).convert('RGB')
        if self.transforms is not None:
            day, night = self.transforms(day), self.transforms(night)
         
        return (day - 0.5) * 2, (night - 0.5) * 2

    def __len__(self):
        return max(len(self.day), len(self.night))