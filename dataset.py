import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image,ImageFile
from torch.utils.data import DataLoader
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataSet(Dataset):
    def __init__(self,args):
        super(DataSet,self).__init__()
        self.dir = args.data_dir
        self.label_file = args.label_dir
        self.im_names = list()
        self.label = list()
        self.trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.dis_type = None
        self.crop_num = args.crop_num
        with open(self.label_file,'r') as f:
            im_label = f.readlines()
            for i in im_label:
                warnings.resetwarnings()
                im_name = i.split(' ')[0]
                im_label = i.split(' ')[1]
                if os.path.isfile(os.path.join(self.dir,im_name)):
                    warnings.filterwarnings('error')
                    try:
                        im_tmp = Image.open(os.path.join(self.dir,im_name))
                        w,h = im_tmp.size
                        im_tmp.close()
                    except:
                        continue
                    warnings.resetwarnings()
                    if w>224 and h>224:
                        for i in range(args.crop_num):
                            self.im_names.append(im_name)
                            self.label.append(float(im_label))
                if args.distortion_divided:
                    self.dis_type = [i.split('/')[0] for i in self.im_names]
        #self.label_std= (self.label-np.min(self.label))/(np.max(self.label)-np.min(self.label))
        self.len = len(self.label)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        im_path = os.path.join(self.dir,self.im_names[idx])
        im = Image.open(im_path)
        im = transforms.functional.to_tensor(im)
        if im.size(0)== 1:
           im = im.repeat(3,1,1)
        im = self.trans(im)
        if self.dis_type == None:
            return im,torch.tensor([self.label[idx]])
        else:
            return im,label_ls,torch.tensor([self.dis_type[idx]])


def get_train_dataloader(args):
    dataset_training = DataSet(args)
    lengths = [int(len(dataset_training)*0.8),int(len(dataset_training)*0.2)+1]
    data_train,data_val = torch.utils.data.random_split(dataset_training,lengths)
    train_loader = DataLoader(dataset_training,
                              batch_size = args.batch_size,
                              shuffle = True,
                              num_workers=4)

    val_loader = DataLoader(data_val,
                              batch_size = args.batch_size,
                              shuffle = True,
                              num_workers=4)

    return train_loader,val_loader

class TestDataSet(Dataset):
    def __init__(self,args):
        super(TestDataSet,self).__init__()
        self.dir = args.testdata_dir
        self.label_file = args.testlabel_dir
        self.im_names = list()
        self.label = list()
        self.trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.dis_type = None
        self.crop_num = args.crop_num
        with open(self.label_file,'r') as f:
            im_label = f.readlines()
            for i in im_label:
                warnings.resetwarnings()
                im_name = i.split(' ')[0]
                im_label = i.split(' ')[1]
                if os.path.isfile(os.path.join(self.dir,im_name)):
                    warnings.filterwarnings('error')
                    try:
                        im_tmp = Image.open(os.path.join(self.dir,im_name))
                        w,h = im_tmp.size
                        im_tmp.close()
                    except:
                        continue
                    warnings.resetwarnings()
                    if w>224 and h>224:
                        for i in range(args.crop_num):
                            self.im_names.append(im_name)
                            self.label.append(float(im_label))
                if args.distortion_divided:
                    self.dis_type = [i.split('/')[0] for i in self.im_names]
        #self.label_std= (self.label-np.min(self.label))/(np.max(self.label)-np.min(self.label))
        self.len = len(self.label)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        im_path = os.path.join(self.dir,self.im_names[idx])
        im = Image.open(im_path)
        im = transforms.functional.to_tensor(im)
        if im.size(0)== 1:
           im = im.repeat(3,1,1)
        im = self.trans(im)
        if self.dis_type == None:
            return im,torch.tensor([self.label[idx]])
        else:
            return im,label_ls,torch.tensor([self.dis_type[idx]])


def get_test_dataloader(args):
    dataset_testing = TestDataSet(args)
    test_loader = DataLoader(dataset_testing,
                             batch_size = args.batch_size,
                             shuffle = False,
                             num_workers=4)
    return test_loader

