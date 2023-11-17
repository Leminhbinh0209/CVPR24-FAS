import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.utils.model_zoo as model_zoo
import math
from torch import Tensor
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    }
from copy import deepcopy
from .mixstyle import InstrClassMixStyle

class NormedLogisticRegression(nn.Module):
    def __init__(self, in_dim:int, out_dim:int=1, ):
        super(NormedLogisticRegression, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.reset_parameters()
 
            
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


    def forward(self, x:Tensor):
        x_norm = F.normalize(x, dim=1)
        w_norm =  F.normalize(self.weight, dim=1)
        logit = F.linear(x_norm, w_norm)
        return logit


class Resnet18(nn.Module):
    def __init__(self, pretrained='imagenet', 
                 bn_freeze = False,
                 n_classes=1):
        super(Resnet18, self).__init__()

        self.model = resnet18(True) # pretrained
        if pretrained=='imagenet':
            self.model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        self.num_ftrs = self.model.fc.in_features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = NormedLogisticRegression(self.num_ftrs, n_classes)
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
                    
        if bn_freeze:
            for m in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                m.eval()
                m.train = lambda _: None
        self.dropout = nn.Dropout(0.0)
        
    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))
        
    def forward(self, x, return_z=False):        
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for blockindex, layerblock in enumerate(self.layer_blocks):
            x = layerblock(x)
        avg_x = self.gap(x)
        x = avg_x 
        z = x.view(x.size(0), -1)
        z = self.dropout(z)
        logit = self.fc(z)
        return (z, logit) if return_z else logit
    
if __name__ == "__main__":
    model = Resnet18().cuda()
    input = torch.randn(128, 3, 256, 256).cuda()
    print(model)
    output = model(input)
    print("Output: ", output[1].shape)
    for n, p in model.named_parameters():
        print(n)