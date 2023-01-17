import torch
import torch.nn as nn
from torch.nn import Parameter

class MarginCosineProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, m=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavieruniform(self.weight)

    def forward(self, input, label):
        cosine = input
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.m
        loss = torch.mean(torch.pow(output, 2))
        return loss
        
    def repr(self):
        return self.__class__.name + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', cos_m=' + str(self.cos_m) \
               + ', sin_m=' + str(self.sin_m) \
               + ', threshold=' + str(self.threshold) + ')'