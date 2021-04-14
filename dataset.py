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
        self.size_rate = list()
        Flag = None
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
                im_attribute = np.array(i.split(' ')[2:-1],dtype=float)
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
                            hw = np.around(h/w,4)
                            self.size_rate.append(hw)
                            self.im_names.append(im_name)
                            self.label.append(float(im_label))
                            if Flag == None:
                                self.attribute = im_attribute
                                Flag = 0
                            else:
                                self.attribute = np.vstack((self.attribute,im_attribute))
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
        lab_att = np.append(np.array(self.label[idx]),self.attribute[idx])
        lab_att = np.append(lab_att,self.size_rate[idx])

        return im,torch.tensor(lab_att)


def get_train_dataloader(args):
    dataset_training = DataSet(args)
    lengths = [int(len(dataset_training)*0.8),int(len(dataset_training)*0.2)+1]
    if (sum(lengths) == len(dataset_training)):
        pass
    else:
        lengths = [int(len(dataset_training)*0.8),int(len(dataset_training)*0.2)]
    data_train,data_val = torch.utils.data.random_split(dataset_training,lengths)
    train_loader = DataLoader(dataset_training,
                              batch_size = args.batch_size,
                              pin_memory=True,
                              shuffle = True,
                              drop_last=True,
                              num_workers=10)

    val_loader = DataLoader(data_val,
                              pin_memory=True,
                              batch_size = args.batch_size,
                              shuffle = True,
                              drop_last=True,
                              num_workers=10)

    return train_loader,val_loader

class TestDataSet(Dataset):
    def __init__(self,args):
        super(TestDataSet,self).__init__()
        self.dir = args.testdata_dir
        self.label_file = args.testlabel_dir
        self.im_names = list()
        self.label = list()
        self.size_rate = list()
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
                            self.size_rate.append(np.around(h/w,4))
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
        lab_r = np.append(np.array(self.label[idx]),self.size_rate[idx])
        return im,torch.tensor(lab_r)

def get_test_dataloader(args):
    dataset_testing = TestDataSet(args)
    test_loader = DataLoader(dataset_testing,
                             pin_memory=True,
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last=True,
                             num_workers=10)
    return test_loader

