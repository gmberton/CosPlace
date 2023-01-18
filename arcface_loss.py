import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
        s == scaling factor
        m == angular margin
        num_class == in_features
        feat_dim == out_features
    """
    def __init__(self, feat_dim, num_class, s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        #logits corresponds to "output" variable in cosface_loss.py 
        logits = self.s * (cos_theta + d_theta)
        #loss = F.cross_entropy(logits, y)

        return logits