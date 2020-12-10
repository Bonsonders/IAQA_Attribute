from ignite.metrics.metric import Metric
import numpy as np
from scipy import stats
import torch

class val_metrics(Metric):

    def reset(self):
        self.y_pred = []
        self.y = []

    def update(self,output):
        pred , y = output
        self.y.append(y[0])
        self.y_pred.append([torch.mean(pred[0])])
    
    def compute(self):
        y = np.reshape(np.asarray(self.y),(-1,))
        pred = np.reshape(np.asarray(self.y_pred),(-1,))
        y = np.array([x.cpu() for x in y])
        pred = np.array([x.cpu() for x in pred])
        srocc = stats.spearmanr(y,pred)[0]
        krocc = stats.stats.kendalltau(y,pred)[0]
        plcc = stats.pearsonr(y,pred)[0]
        rmse = np.sqrt(((y - pred) ** 2).mean())

        return srocc,krocc,plcc,rmse