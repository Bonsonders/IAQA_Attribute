import torch
from torch import nn
import torchvision.models as model_template
import torch.nn.functional as F

class IAQA_model(nn.Module):
    def __init__ (self,args):
        super(IAQA_model,self).__init__()
        RESNET = model_template.resnet50(pretrained=True)
        self.features  = nn.Sequential(
            RESNET.conv1,
            RESNET.bn1,
            RESNET.relu,
            RESNET.maxpool,
            RESNET.layer1,
            RESNET.layer2,
            RESNET.layer3,
            RESNET.layer4)

        self.classifer = nn.Sequential(
            nn.Linear(2048, args.layer_num),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(args.layer_num),
            nn.Linear(args.layer_num, args.layer_num),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(args.layer_num),
            nn.Linear(args.layer_num,1)
            )

    def forward(self,x):
        x= self.features(x)
        x= F.adaptive_avg_pool2d(x,(x.size(-2),x.size(-1)))
        x= x.view(x.size(0),-1)
        x= self.classifer(x)
        return x

