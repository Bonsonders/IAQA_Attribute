from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import os
import sys
import torch 


class TrainOptions():

    def __init__(self):
        super(TrainOptions,self).__init__()
    
    def initialize(self,parser):
        parser.add_argument('--lr',type=float,default=0.1,help='initial learning rate')
        parser.add_argument('--epochs',type=int,default=200,help='number of ephoes to train (default:200)')
        parser.add_argument('--gpu',type=bool,default=True,help='flag whether to use GPU acceleration')
        parser.add_argument('--batch_size',type=int,default=32,help='input batchsize for training(default:32)')
        parser.add_argument('--log_interval)',type=int,default=10,help='number of interval to report the status')
        parser.add_argument('--label_dir',type=str,default='',help='directory for label.txt/label.csv')
        parser.add_argument('--data_dir',type=str,default='',help='directory for Dataset')
    
    
    def parse(self):
        parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser) 
        self.parser = parser
        
        
        return parser.parse_args()

    def print_opt(self,opt):
        message = ''
        message += '----------------- Opts ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-----------------------------------------'
        print(message)

        # save
        exp_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        file_name = os.path.join(exp_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        