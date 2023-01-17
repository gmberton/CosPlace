import torch
import torch.nn as nn
from torch.nn import Parameter

class MarginCosineProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, margin=0.5, num_classes=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.num_classes = num_classes
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavieruniform(self.weight)

    def forward(self, input, label):
        cosine = input
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        other_logits = torch.cos(theta)
        output = torch.where(one_hot > 0, target_logits, other_logits)
        loss = torch.mean(torch.pow(output, 2))
        return loss
        
    def repr(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', margin=' + str(self.margin) \
               + ', num_classes=' + str(self.num_classes) + ')'