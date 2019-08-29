from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
from PIL import Image
import os
import numpy as np
import rawpy

class trainset(Dataset):
    def __init__(self, root, root_out):
        #self.images = sorted(glob.glob(root + '/*.*'))
        self.target = sorted(glob.glob(root_out + '/*.*'))
        train_fns = glob.glob(root_out + '0*.ARW')
        self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
        self.file_probs = self.parse_prob()
        self.input_dir = root
        self.gt_dir = root_out
        
    def parse_prob(self):
        f = open('./prob.txt', 'r')
        file_prob = {}
        f1 = f.readlines()
        for x in f1:
            [idx, prob_list] = x.split(" ")
            p1, p2, p3, _ = prob_list.split(",")
            file_list = ['0.1', '0.04', '0.033']
            prob_sum = float(int(p1) + int(p2) + int(p3))
            file_prob[int(idx)] = [int(p1)/prob_sum, int(p2)/prob_sum, int(p3)/prob_sum]
        return file_prob
        
    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out
        
            

    def __getitem__(self, index):
        train_id = self.train_ids[index]
        prob = self.file_probs[int(train_id)]
        in_files = ['0.1', '0.04', '0.033']
        sec = np.random.choice(in_files, 1, p = prob)[0]
        in_path = self.input_dir + '%05d_00_%ss.ARW' %( train_id , sec )
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300) 
        
        raw = rawpy.imread(in_path)
        img = self.pack_raw(raw) * ratio

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        out_img = np.float32(im / 65535.0)
        
        return {'lr': img, 'hr': out_img}

    def __len__(self):
        return len(self.target)
