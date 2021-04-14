import torch
from torch import nn
import torchvision.models as model_template
import torch.nn.functional as F
import warnings

def insert_inplate_1_2(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if m.kernel_size==(1,1):
            m.dilation=(1,2)
        if m.kernel_size==(3,3):
            m.dilation=(1,2)
            m.padding=(1,2)

def insert_inplate_2_1(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if m.kernel_size==(1,1):
            m.dilation=(2,1)
        if m.kernel_size==(3,3):
            m.dilation=(2,1)
            m.padding=(2,1)

def AFDC(rate):
    warnings.simplefilter("ignore")
    result = None
    for i in rate:
        a = torch.ceil(i)-torch.tensor(i)
        b = torch.tensor(i)-torch.floor(i)
        if 0.5<i<1:
            tmp = torch.tensor([0,a,b])
        elif 1<i<2:
            tmp = torch.tensor([a,b,0])
        else:
            tmp = torch.tensor([0,1,0])
        if result == None:
            result = tmp
        else:
            result = torch.vstack((result,tmp))
    warnings.resetwarnings()
    return result.float()


class IAQA_model(nn.Module):
    def __init__ (self,args):
        super(IAQA_model,self).__init__()
        self.args = args
        MobileNet = model_template.mobilenet_v3_small(pretrained=True)
        self.features1 = MobileNet.features.apply(insert_inplate_1_2)
        self.features2 = MobileNet.features
        self.features3 = MobileNet.features.apply(insert_inplate_2_1)
        self.classifer1 = nn.Sequential(
            nn.Linear(576*3, args.layer_num),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(args.layer_num,1)
            )
        self.classifer2 = nn.Sequential(
                    nn.Linear(960,args.layer_num),
                    nn.LeakyReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(args.layer_num,3)
        )

    def forward(self,x,r):
        x= x.float()
        out1= self.features1(x)
        out2= self.features2(x)
        out3= self.features3(x)
        out1= F.avg_pool2d(out1,(out1.size(-2),out1.size(-1)))
        out2= F.avg_pool2d(out2,(out2.size(-2),out2.size(-1)))
        out3= F.avg_pool2d(out3,(out3.size(-2),out3.size(-1)))
        out1= out1.view(out1.size(0),-1)
        out2= out2.view(out2.size(0),-1)
        out3= out3.view(out3.size(0),-1)
        ratio = AFDC(r)
        out1 = ratio[:,0].unsqueeze(1).cuda()*out1
        out2 = ratio[:,1].unsqueeze(1).cuda()*out2
        out3 = ratio[:,2].unsqueeze(1).cuda()*out3
        out= torch.cat((out1,out2,out3),1)
        out= self.classifer1(out)
        return out

