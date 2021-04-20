import torch
import time
from torch.utils.data import DataLoader
from utils.options import TestOptions
import numpy as np
from dataset import get_test_dataloader
from models import *
from utils.metrics import evaluate
from scipy import stats


def test(l_list,t_list):
    start_time = time.time()
    test_loader = get_test_dataloader(args)
    for batch_index,(ims,labels) in enumerate(test_loader):
        if args.gpu:
            ims = ims.cuda()
            labels = labels.cuda()
        size_ratio = labels[:,-1]
        outs,att = model(ims,size_ratio)
        l_list = np.append(l_list,labels[:,0:1].detach().cpu().numpy())
        t_list = np.append(t_list,outs.detach().cpu().numpy())

    end_time = time.time()
    time_cost = end_time - start_time
    return time_cost,l_list,t_list

if __name__ == '__main__':
    args = TestOptions().parse()
    model = IAQA_model(args)
    weight = torch.load(args.weights)
    model.load_state_dict(weight)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_l = np.array([])
    y_p = np.array([])

    time_cost,y_l,y_p = test(y_l,y_p)
    SROCC, KROCC, PLCC, RMSE, Acc = evaluate(y_l,y_p)
    print("Testing Results - Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f}"
             .format(Acc, RMSE,SROCC,KROCC,PLCC))
    print("TIME COST:",time_cost)
