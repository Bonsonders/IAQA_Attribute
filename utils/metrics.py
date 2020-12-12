from ignite.metrics.metric import Metric
import numpy as np
from scipy import stats
import torch

class val_metrics(Metric):

    def reset(self):
        self.y_p = list()
        self.y = list()

    def update(self,output):
        pred,y = output
        self.y.append(y[0])
        self.y_p.append([torch.mean(pred[0])])
    
    def compute(self):
        y = np.reshape(np.asarray(self.y),(-1,))
        pred = np.reshape(np.asarray(self.y_p),(-1,))
        y = np.array([x.cpu() for x in y])
        pred = np.array([x.cpu() for x in pred])
        #Calulate the Metric of our mission
        SROCC = stats.spearmanr(y,pred)[0]
        KROCC = stats.stats.kendalltau(y,pred)[0]
        PLCC = stats.pearsonr(y,pred)[0]
        RMSE = np.sqrt(((y - pred) ** 2).mean())
        ACC = (y == pred).mean()

        return SROCC,KROCC,PLCC,RMSE,ACC