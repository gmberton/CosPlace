import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    """reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
       reference2: <Additive Margin Softmax for Face Verification>

       CosFace implementation from: https://github.com/ydwen/opensphere/tree/main/model/head
    """
    def __init__(self, in_features, out_features, s=64., m=0.35):
        super(CosFace, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            d_theta = torch.zeros_like(cos_theta)
            d_theta.scatter_(1, y.view(-1, 1), -self.m, reduce='add')

        #logits corresponds to "output" variable in cosface_loss.py 
        logits = self.s * (cos_theta + d_theta)
        #loss = F.cross_entropy(logits, y)

        return logits