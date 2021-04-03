from ignite.metrics.metric import Metric
import numpy as np
from scipy import stats
import torch

class val_metrics(Metric):

    def reset(self):
        self.y = np.array([])
        self.y_p = np.array([])

    def update(self,output):
        pred,y = output
        self.y = np.append(self.y,y.cpu().numpy())
        self.y_p = np.append(self.y_p,pred.cpu().numpy())

    def compute(self):
        y = np.reshape(np.asarray(self.y),(-1,))
        pred = np.reshape(np.asarray(self.y_p),(-1,))
        #Calulate the Metric of our mission
        SROCC = stats.spearmanr(y,pred)[0]
        KROCC = stats.stats.kendalltau(y,pred)[0]
        PLCC = stats.pearsonr(y,pred)[0]
        RMSE = np.sqrt(((y - pred) ** 2).mean())
        ACC = (y == pred).mean()

        return SROCC,KROCC,PLCC,RMSE,ACC


class test_metrics(Metric):
    def __init__(self,args):
        self.num = args.crop_num

    def reset(self):
        self.y = np.array([])
        self.y_p = np.array([])

    def update(self,output):
        pred,y = output
        self.y = np.append(self.y,y.cpu().numpy()[::self.num])
        predd = pred.cpu().numpy().reshape(-1,self.num)
        self.y_p = np.append(np.mean(predd,axis = -1))

    def compute(self):
        y = np.reshape(np.asarray(self.y),(-1,))
        pred = np.reshape(np.asarray(self.y_p),(-1,))
        #Calulate the Metric of our mission
        SROCC = stats.spearmanr(y,pred)[0]
        KROCC = stats.stats.kendalltau(y,pred)[0]
        PLCC = stats.pearsonr(y,pred)[0]
        RMSE = np.sqrt(((y - pred) ** 2).mean())
        ACC = (y == pred).mean()

        return SROCC,KROCC,PLCC,RMSE,ACC
