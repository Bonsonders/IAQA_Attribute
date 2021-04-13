import torch
from torch import nn
import torchvision.models as model_template
import torch.nn.functional as F
from utils.mobilenetv3 import *

class IAQA_model(nn.Module):
    def __init__ (self,args):
        super(IAQA_model,self).__init__()
        self.args = args
        MobileNet = mobilenet_v3_small(pretrained=True)
        self.features  = MobileNet.features
        #RESNET = model_template.resnet50(pretrained=True)
       #self.features  = nn.Sequential(
       #                 RESNET.conv1,
       #                 RESNET.bn1,
       #                 RESNET.relu,
       #                 RESNET.maxpool,
       #                 RESNET.layer1,
       #                 RESNET.layer2,
        #                RESNET.layer3,
       #                 RESNET.layer4)
        self.classifer1 = nn.Sequential(
            nn.Linear(576, args.layer_num),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(args.layer_num,1)
            )
        #self.classifer2 = nn.Sequential(
        #            nn.Linear(960, 512),
        #            nn.LeakyReLU(True),
        #            nn.Dropout(0.5),
        #            nn.Linear(512,3)
        #)

    def forward(self,x):
        x= x.float()
        x= self.features(x)
        x= F.avg_pool2d(x,(x.size(-2),x.size(-1)))
        x= x.view(x.size(0),-1)
        x1= self.classifer1(x)
        if self.args.attribute:
            x2= self.classifer2(x)
            return x1,x2
        else:
            return x1

