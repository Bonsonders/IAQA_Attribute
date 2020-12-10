import torch
import os
from torch.utils.data import DataLoader
from utils.options import TrainOptions
import numpy as np 
from dataset import DataSet
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from utils.metrics import val_metrics
from models import *


if __name__ == "__main__":

    args = TrainOptions().parse()
    model = None #TODO
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9,weight_decay=5e-4)
    val_metric = val_metrics()
    criterion = torch.nn.L1Loss()
    
    
    train_loader = DataLoader(DataSet(args),batch_size = args.batch_size,shuffle = True)
    val_loader = None #TODO
    
    trainer = create_supervised_trainer(model,optimizer,criterion,device = device)
    
    evaluator = create_supervised_evaluator(model,metrics = val_metric,device = device)
    @trainer.on(Events.ITERATION_COMPLETED(every = args.log_interval))
    
    
    def training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.ITERATION_COMPLETED)
    def training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.5f}".format(trainer.state.epoch, metrics["acc"], metrics["loss"]))

    @trainer.on(Events.ITERATION_COMPLETED)
    def Validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.5f}".format(trainer.state.epoch, metrics["acc"], metrics["loss"]))


    trainer.run(train_loader,max_epochs = args.epochs)




