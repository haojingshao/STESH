import torch
import torch.nn
import torch.nn.functional as F
from utils import cosine_similarity
import numpy as np


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6)) # type: ignore
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = torch.where(torch.isnan(final), torch.zeros_like(final) + np.inf, final)
        if mean:
            final = torch.mean(final)
        return final



def regularization_loss(emb, graph_nei_sadj, graph_neg_sadj,graph_nei_madj, graph_neg_madj):
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    sadj_neigh_loss = torch.mul(graph_nei_sadj, torch.log(mat)).mean() 
    sadj_neg_loss = torch.mul(graph_neg_sadj, torch.log(1 - mat)).mean() 
    sadj_pair_loss = -(sadj_neigh_loss + sadj_neg_loss) / 2 
    madj_neigh_loss = torch.mul(graph_nei_madj, torch.log(mat)).mean() 
    madj_neg_loss = torch.mul(graph_neg_madj, torch.log(1 - mat)).mean() 
    madj_pair_loss = -(madj_neigh_loss + madj_neg_loss) / 2 
    regularization_loss= sadj_pair_loss + madj_pair_loss
    return regularization_loss


def consistency_loss(emb1, emb2 ,emb3):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb3 = emb3 - torch.mean(emb3, dim=0, keepdim=True)
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    emb3 = F.normalize(emb3, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cov3 = torch.matmul(emb3, emb3.t())
    consistency_loss = torch.mean((cov1 - cov2 ) ** 2 + (cov1 - cov3 ) ** 2 + (cov2 - cov3 ) ** 2) 
    return consistency_loss