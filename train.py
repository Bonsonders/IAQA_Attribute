import os
import torch
from utils.options import TrainOptions
import numpy as np
import torch.nn.functional as F
from dataset import get_test_dataloader,get_train_dataloader
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from utils.metrics import val_metrics
from models import *
from scipy import stats
import adabound

if __name__ == "__main__":

    Option = TrainOptions()
    args = Option.parse()
    Option.print_opt()
    model = IAQA_model(args)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9,weight_decay=5e-4)
    optimizer = adabound.AdaBound(model.parameters(), lr=args.lr, final_lr=0.1) #Adabound: Adaboost+ SGD
    val_metric = val_metrics()
    criterion = torch.nn.L1Loss()
    tensorboard_dir = os.path.join(args.runs,args.name)
    writer = SummaryWriter(tensorboard_dir)
    checkpoint_path = os.path.join(args.checkpoints_dir,args.name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoints = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


    train_loader,val_loader = get_train_dataloader(args)
    test_loader = get_test_dataloader(args)
    trainer = create_supervised_trainer(model, optimizer, criterion, device= device)
    global best_criterion
    best_criterion = -1
    evaluator = create_supervised_evaluator(model,metrics = {'val': val_metric},device = device)
    #writer.add_graph(model)
    global lr
    lr = args.lr


    @trainer.on(Events.ITERATION_COMPLETED(every = args.log_interval))
    def training_loss(trainer):
        writer.add_scalar("Train/loss", trainer.state.output, trainer.state.iteration)
        #print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def training_results(trainer):
        global lr
        print("====================Epoch:{}==================== Learning Rate:{:.5f}".format(trainer.state.epoch,lr))
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, Acc = metrics['val']
        print("Training Results - Epoch: {}  Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f}"
             .format(trainer.state.epoch, Acc, RMSE,SROCC,KROCC,PLCC))
        if trainer.state.epoch % 20 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
                lr = p['lr']


    @trainer.on(Events.EPOCH_COMPLETED)
    def Validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, Acc = metrics['val']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f}"
            .format(trainer.state.epoch, Acc, RMSE,SROCC,KROCC,PLCC))
        writer.add_scalar('Val/LOSS', RMSE, trainer.state.epoch)
        writer.add_scalar('Val/SROCC', SROCC, trainer.state.epoch)
       # global best_criterion
       # if SROCC > best_criterion:
           #torch.save(model.state_dict(), checkpoints.format(net=args.name, epoch=trainer.state.epoch, type='val'))
           # best_criterion = SROCC
           # writer.add_text('Best_Criertion',"Epoch:{} {:.5f}".format(trainer.state.epoch,best_criterion))

        if args.distortion_divided:
            labels = list()
            preds = list()
            dis_types = list()
            for (im, label, dis_type) in val_loader:
                labels.append({dis_type:label})
                dis_types.append(dis_type)
                if args.gpu:
                    im= im.cuda()
                    label = label.cuda()
                pred = model(im)
                preds.append({dis_type:np.asarray(pred)})
            dis = list(set(dis_types))
            for i in dis:
                l_tmp = [j[i] for j in labels if i in j]
                preds_tmp = [j[i] for j in preds if i in j]
                SROCC = stats.spearmanr(l_tmp,preds_tmp)[0]
                RMSE = np.sqrt(((l_tmp - preds_tmp) ** 2).mean())
                writer.add_scalar('{}/LOSS'.format(i),RMSE,trainer.state.epoch)
                writer.add_scalar('{}/SROCC'.format(i),SROCC,trainer.state.epoch)
                print("Distortion_Type:{} Test Results - RMSE: {:.5f} SROCC: {:.5f}"
                    .format(i, RMSE, SROCC))



    @trainer.on(Events.EPOCH_COMPLETED)
    def Test_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, Acc = metrics['val']
        print("Testing Results - Epoch: {}  Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f}"
            .format(trainer.state.epoch, Acc, RMSE,SROCC,KROCC,PLCC))
        writer.add_scalar('Test/LOSS', RMSE, trainer.state.epoch)
        writer.add_scalar('Test/SROCC', SROCC, trainer.state.epoch)
        global best_criterion
        if abs(SROCC) > best_criterion:
            torch.save(model.state_dict(), checkpoints.format(net=args.name, epoch=trainer.state.epoch, type='test'))
            best_criterion = SROCC
            writer.add_text('Best_Criertion',"Epoch:{} {:.5f}".format(trainer.state.epoch,best_criterion))

        if args.distortion_divided:
            labels = list()
            preds = list()
            dis_types = list()
            for (im, label, dis_type) in test_loader:
                labels.append({dis_type:label})
                dis_types.append(dis_type)
                if args.gpu:
                    im= im.cuda()
                    label = label.cuda()
                pred = model(im)
                preds.append({dis_type:np.asarray(pred)})
            dis = list(set(dis_types))
            for i in dis:
                l_tmp = [j[i] for j in labels if i in j]
                preds_tmp = [j[i] for j in preds if i in j]
                SROCC = stats.spearmanr(l_tmp,preds_tmp)[0]
                RMSE = np.sqrt(((l_tmp - preds_tmp) ** 2).mean())
                writer.add_scalar('{}/LOSS'.format(i),RMSE,trainer.state.epoch)
                writer.add_scalar('{}/SROCC'.format(i),SROCC,trainer.state.epoch)
                print("Distortion_Type:{} Test Results - RMSE: {:.5f} SROCC: {:.5f}"
                    .format(i, RMSE, SROCC))



    trainer.run(train_loader, max_epochs= args.epochs)
