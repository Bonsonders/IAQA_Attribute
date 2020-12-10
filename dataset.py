import torch
import numpy as np
import os
from torch.utils.data import Dataset 
from PIL import Image


class DataSet(Dataset):
    def __init__(self,args):
        super(DataSet,self).__init__()
        self.dir = args.data_dir
        self.label_file = args.label_dir
        self.im_names = list()
        self.label = list()
        with open(self.label_file,'r') as f:
            im_label = f.readlines()
            for i in im_label:
                self.im_names.append(i.split(' ')[0])
                self.label.append(float(i.split(' ')[1]))
        self.len = len(self.label)
    
    def __len__(self):
        return self.len
  
    def __getiem__(self,idx):
        im_path = os.path.join(self.dir,self.im_names[idx])
        im = Image.open(im_path)
        return im,self.label[idx]






