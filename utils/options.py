from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import os
import sys
import torch
from datetime import datetime


class TrainOptions():

    def __init__(self):
        super(TrainOptions,self).__init__()

    def initialize(self,parser):
        parser.add_argument('--lr',type= float,default=0.1,help='initial learning rate')
        parser.add_argument('--epochs',type= int,default=200,help='number of ephoes to train (default:200)')
        parser.add_argument('--gpu',type= bool,default=True,help='flag whether to use GPU acceleration')
        parser.add_argument('--batch_size',type= int,default=32,help='input batchsize for training(default:32)')
        parser.add_argument('--log_interval',type= int,default=10,help='number of interval to report the status')
        parser.add_argument('--label_dir',type= str,default='',help='directory for label.txt/label.csv')
        parser.add_argument('--data_dir',type= str,default='',help='directory for Dataset')
        parser.add_argument('--checkpoints_dir', type= str, default='./checkpoints', help='models save directory')
        parser.add_argument('--runs',type= str,default='./runs',help='tenserboardX directory')
        parser.add_argument('--name',type= str,default='expriment',help='expriment name')
        parser.add_argument('--distortion_divided',action='store_true',default=False,help='Test the dataset for seperate type of distortion')
        parser.add_argument('--layer_num',type=int,default=512, help='number of inner layer')
        parser.add_argument('--crop_num',type= int, default=2, help='set the number of crop')
        parser.add_argument('--testlabel_dir',type= str,default='',help='directory for test dataset label.txt/label.csv')
        parser.add_argument('--testdata_dir',type= str,default='',help='directory for test Dataset')
        return parser

    def parse(self):
        parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        self.parser = parser
        self.opt = parser.parse_args()
        return parser.parse_args()

    def print_opt(self):
        message = ''
        message += '----------------- Opts ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '---------------------------------------'
        print(message)

        # save
        exp_dir = os.path.join(self.opt.checkpoints_dir,self.opt.name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        now = datetime.now()
        time_now = now.strftime("%d-%m-%Y-%H-%M")
        file_name = os.path.join(exp_dir, '{}_opt.txt'.format(time_now))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

class TestOptions():
    def __init__(self):
        super(TestOptions,self).__init__()

    def initialize(self,parser):
        parser.add_argument('--gpu',type=bool,default=True,help='flag whether to use GPU acceleration')
        parser.add_argument('--batch_size',type=int,default=32,help='input batchsize for training(default:32)')
        parser.add_argument('--net',type=str,required=True,help='net model you want to test')
        parser.add_argument('--weights',type=str,required=True,help='the weights file you want to test')
        parser.add_argument('--Dataset',type=str,required=True,help='the dataset you want to test')
        parser.add_argument('--distortion_divided',action='store_true',default=False,help='Test the dataset for seperate type of distortion')
        return parser

    def parse(self):
        parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        return parser.parse_args()
