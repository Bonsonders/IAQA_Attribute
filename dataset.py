import torch
import numpy as np
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader


class DataSet(Dataset):
    def __init__(self,args):
        super(DataSet,self).__init__()
        self.dir = args.data_dir
        self.label_file = args.label_dir
        self.im_names = list()
        self.label = list()
        self.trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(224)])
        self.dis_type = None
        with open(self.label_file,'r') as f:
            im_label = f.readlines()
            for i in im_label:
                self.im_names.append(i.split(' ')[0])
                self.label.append(float(i.split(' ')[1]))
                if args.distortion_divided:
                    self.dis_type = [i.split('/')[0] for i in self.im_names]
        self.label_std= (self.label-np.min(self.label))/(np.max(self.label)-np.min(self.label))
        self.len = len(self.label_std)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        im_path = os.path.join(self.dir,self.im_names[idx])
        im = Image.open(im_path)
        im = self.trans(im)
        if im.size(0)== 1:
           im = im.repeat(3,1,1)
        if self.dis_type == None:
            return im,torch.tensor([self.label[idx]])
        else:
            return im,torch.tensor([self.label[idx]]),torch.tensor([self.dis_type[idx]])


def get_train_dataloader(args):


    dataset_training = DataSet(args)
    lengths = [int(len(dataset_training)*0.8),int(len(dataset_training)*0.2)]
    data_train,data_val = torch.utils.data.random_split(dataset_training,lengths)
    train_loader = DataLoader(data_train,
                              batch_size = args.batch_size,
                              shuffle = True,
                              num_workers=4)

    val_loader = DataLoader(data_val,
                              batch_size = args.batch_size,
                              shuffle = True,
                              num_workers=4)

    return train_loader,val_loader

def get_test_dataloader(args):
    dataset_testing = DataSet(args)
    test_loader = DataLoader(dataset_testing,
                             batch_size = args.batch_size,
                             shuffle = True,
                             num_workers=4)
    return test_loader

