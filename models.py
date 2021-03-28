import torch
from torch import nn
import torchvision.models as model_template
import torch.nn.functional as F

class IAQA_model(nn.Module):
    def __init__ (self,args):
        super(IAQA_model,self).__init__()
        VGG = model_template.vgg16(pretrained=True)
        self.features  = VGG.features
        self.classifer = nn.Sequential(
            nn.Linear(512, args.layer_num),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(args.layer_num),
            nn.Linear(args.layer_num, args.layer_num),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(args.layer_num),
            nn.Linear(args.layer_num,1)
            )

    def forward(self,x):
        x= self.features(x)
        x= F.avg_pool2d(x,(x.size(-2),x.size(-1)))
        x= x.view(x.size(0),-1)
        x= self.classifer(x)
        return x

