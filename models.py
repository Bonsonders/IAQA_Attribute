import torch
from torch import nn
import torchvision.models as model_template
import torch.nn.functional as F

class IAQA_model(nn.Module):
    def __init__ (self,args):
        super(IAQA_model,self).__init__()
        MobileNet = model_template.mobilenet_v3_large(pretrained=False)
        self.features  = MobileNet.features
        self.classifer = nn.Sequential(
            nn.Linear(960, args.layer_num),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(args.layer_num,1)
            )

    def forward(self,x):
        x= self.features(x)
        x= F.avg_pool2d(x,(x.size(-2),x.size(-1)))
        x= x.view(x.size(0),-1)
        x= self.classifer(x)
        return x

