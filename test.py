import torch
from torch.utils.data import DataLoader
from utils.options import TestOptions
import numpy as np
from dataset import get_test_dataloader
from models import *
from ignite.engine import create_supervised_evaluator
from utils.metrics import val_metrics
from scipy import stats

if __name__ == '__main__':
    args = TestOptions().parse()
    model = None 
    weight = torch.load(args.weights)
    model.load_state_dict(weight).eval()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    val_metric = val_metrics()
    val_loader = get_test_dataloader(args) 

    evaluator = create_supervised_evaluator(model,metrics = {'val': val_metric},device = device)
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    SROCC, KROCC, PLCC, RMSE, Acc = metrics['val']
    print("Test Results - Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f}"
            .format(Acc,RMSE,SROCC,KROCC,PLCC))
    
    
    if args.distortion_divided:
        labels = list()
        preds = list()
        dis_types = list()
        for (im,label,dis_type) in val_loader:
            labels.append({dis_type:label})
            dis_types.append(dis_type)
            if args.gpu:
                im  = im.cuda()
                label = label.cuda()
            pred = model(im)
            preds.append({dis_type:np.asarray(pred)})
        dis = list(set(dis_types))
        for i in dis:
            l_tmp = [j[i] for j in labels if i in j]
            preds_tmp = [j[i] for j in preds if i in j]
            SROCC = stats.spearmanr(l_tmp,preds_tmp)[0]
            RMSE = np.sqrt(((l_tmp - preds_tmp) ** 2).mean())
            print("Distortion_Type:{} Test Results - RMSE: {:.5f} SROCC: {:.5f}"
                .format(i,RMSE,SROCC))


    