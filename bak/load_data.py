from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
from PIL import Image
import os
import numpy as np

class trainset(Dataset):
    def __init__(self, root, root_out):
        self.images = sorted(glob.glob(root + '/*.*'))
        self.target = sorted(glob.glob(root_out + '/*.*'))


    def __getitem__(self, index):
        try:
            img = np.asarray(Image.open(self.images[index % len(self.images)]))
        except Exception as e:
            index = index - 1 if index > 0 else index + 1
            print(str(index) + '!!!!!!!!!!!!!!!!!!!!!!!!!')
            img = np.asarray(Image.open(self.images[index % len(self.images)]))
            

        try:
            out_img = np.asarray(Image.open(self.target[index % len(self.target)]))
        except Exception as e:
            index = index - 1 if index > 0 else index + 1
            print('canon---' + str(index) + '!!!!!!!!!!!!!!!!!!!!!!!!!')
            out_img = np.asarray(Image.open(self.target[index % len(self.target)]))
         
        return {'lr': img, 'hr': out_img}

    def __len__(self):
        return len(self.images)
