import os
import torch
import time
from utils.options import TrainOptions
import numpy as np
import torch.nn.functional as F
from dataset import get_test_dataloader,get_train_dataloader
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from utils.metrics import val_metrics,test_metrics,evaluate
from models import *
from scipy import stats
import adabound
from utils.regress_loss import RegressionLoss

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def train(epoch,l_list,t_list):
    start_time = time.time()
    for batch_index,(ims,labels) in enumerate(train_loader):
        if args.gpu:
            ims = ims.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()
        size_ratio = labels[:,-1]
        if args.attribute:
            outs,att = model(ims,size_ratio)
            loss2 = torch.nn.functional.l1_loss(att,labels[:,1:-1])
            loss2.backward(retain_graph=True)
        else:
            outs = model(ims,size_ratio)
        loss = criterion(outs.float(),labels[:,0:1].float())
        loss.backward(retain_graph=True)

        optimizer.step()
        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1
        writer.add_scalar("Train/loss", loss.item(),n_iter)

        l_list = np.append(l_list,labels[:,0:1].detach().cpu().numpy())
        t_list = np.append(t_list,outs.detach().cpu().numpy())

    end_time = time.time()
    time_cost = end_time - start_time
    return time_cost,l_list,t_list


def test(l_list,t_list):
    for batch_index,(ims,labels) in enumerate(test_loader):
        if args.gpu:
            ims = ims.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        size_ratio = labels[:,-1]
        if args.attribute:
            outs,att = model(ims,size_ratio)
        else:
            outs = model(ims,size_ratio)
        l_list = np.append(l_list,labels[:,0:1].detach().cpu().numpy())
        t_list = np.append(t_list,outs.detach().cpu().numpy())

    return l_list,t_list


if __name__ == "__main__":

    Option = TrainOptions()
    args = Option.parse()
    Option.print_opt()
    model = IAQA_model(args)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.apply(inplace_relu)
    #optimizer = torch.optim.Adam(model.parameters(),lr = args.lr, weight_decay = 1e-3)
    #optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9,weight_decay=5e-4)
    optimizer = adabound.AdaBound(model.parameters(), lr=args.lr, final_lr=0.1) #Adabound: Adaboost+ SGD
    criterion = RegressionLoss()
    tensorboard_dir = os.path.join(args.runs,args.name)
    writer = SummaryWriter(tensorboard_dir)
    checkpoint_path = os.path.join(args.checkpoints_dir,args.name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoints = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_criterion = -1
    lr = args.lr
    train_loader,val_loader = get_train_dataloader(args)
    test_loader = get_test_dataloader(args)


    for epoch in range(1,args.epochs+1):

        y_l = np.array([])
        y_p = np.array([])
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        time_cost, y_l, y_p = train(epoch,y_l,y_p)
        print("====================Epoch:{}==================== Learning Rate:{:.5f}".format(epoch,current_lr))
        SROCC, KROCC, PLCC, RMSE, Acc = evaluate(y_l,y_p)
        writer.add_scalar('Train/SROCC', SROCC,epoch)
        print("Training Results - Epoch: {}  Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f} ***** Time Cost: {:.1f} s"
             .format(epoch, Acc, RMSE,SROCC,KROCC,PLCC,time_cost))


        y_l = np.array([])
        y_p = np.array([])
        start = time.time()
        y_l, y_p = test(y_l,y_p)
        SROCC, KROCC, PLCC, RMSE, Acc = evaluate(y_l,y_p)
        end = time.time()
        writer.add_scalar('Test/LOSS', RMSE,epoch)
        writer.add_scalar('Test/SROCC', SROCC,epoch)
        print("Testing Results - Epoch: {}  Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f} ***** Time Cost: {:.1f} s"
             .format(epoch, Acc, RMSE,SROCC,KROCC,PLCC,end-start))
        if abs(SROCC) > best_criterion:
            torch.save(model.state_dict(), checkpoints.format(net=args.name, epoch=epoch, type='test'))
            best_criterion = abs(SROCC)
            writer.add_text('Best_Criertion',"Epoch:{} {:.5f}".format(epoch,best_criterion))
#------------------------------------------------------------------------#

            ####################################################
            #                                                  #
            #          This is Generate by Ignite              #
            #        A Tools for training and testing          #
            #           For more please refer to:              #
            #                                                  #
            #            https://pytorch.org/ignite/           #
            #                                                  #
            ####################################################

#------------------------------------------------------------------------#
'''
    trainer = create_supervised_trainer(model, optimizer, criterion, device= device)
    val_metric = val_metrics()
    test_metric = test_metrics()
    test_metric.crop_num(args)
    evaluator = create_supervised_evaluator(model,metrics = {'val': val_metric},device = device)

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
        writer.add_scalar('Train/SROCC', SROCC, trainer.state.epoch)
        print("Training Results - Epoch: {}  Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f}"
             .format(trainer.state.epoch, Acc, RMSE,SROCC,KROCC,PLCC))


    @trainer.on(Events.EPOCH_COMPLETED)
    def Validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, Acc = metrics['val']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f}"
            .format(trainer.state.epoch, Acc, RMSE,SROCC,KROCC,PLCC))
        writer.add_scalar('Val/LOSS', RMSE, trainer.state.epoch)
        writer.add_scalar('Val/SROCC', SROCC, trainer.state.epoch)

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
        test_metric.attach(evaluator,"test")
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, Acc = metrics['test']
        test_metric.detach(evaluator)
        print("Testing Results - Epoch: {}  Avg accuracy: {:.3f} RMSE: {:.5f}  SROCC: {:.5f} KROCC: {:.5f} PLCC: {:.5f}"
            .format(trainer.state.epoch, Acc, RMSE,SROCC,KROCC,PLCC))
        writer.add_scalar('Test/LOSS', RMSE, trainer.state.epoch)
        writer.add_scalar('Test/SROCC', SROCC, trainer.state.epoch)
        global best_criterion
        if abs(SROCC) > best_criterion:
            torch.save(model.state_dict(), checkpoints.format(net=args.name, epoch=trainer.state.epoch, type='test'))
            best_criterion = abs(SROCC)
            writer.add_text('Best_Criertion',"Epoch:{} {:.5f}".format(trainer.state.epoch,best_criterion))


    trainer.run(train_loader, max_epochs= args.epochs)
'''
#------------------------------------------------------------------------#
