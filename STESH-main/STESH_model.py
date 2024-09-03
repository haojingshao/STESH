import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from decoder import decoder
from attention import Attention


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout): 
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(nfeat, nhid)
        self.gcn2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj)) 
        x = F.dropout(x, self.dropout, training=self.training)  
        x = self.gcn2(x, adj) 
        return x
    


class STESH(nn.Module): 
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(STESH, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.MGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.NB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )

    def forward(self, x, sadj, fadj , madj):
        emb1 = self.SGCN(x, sadj)  # Spatial_GCN
        com1 = self.CGCN(x, sadj)  # Co_GCN
        emb2 = self.FGCN(x, fadj)  # Feature_GCN
        com2 = self.CGCN(x, fadj)  # Co_GCN
        emb3 = self.MGCN(x, madj)  # morphological_GCN
        com3 = self.CGCN(x, madj)  # Co_GCN
        
        emb = torch.stack([emb1,  emb2 , emb3 ,(com1 + com2 + com3) / 3], dim=1) 
        emb, att = self.att(emb) 
        emb = self.MLP(emb) 
        [disp, mean] = self.NB(emb) 
        return com1, com2, com3 , emb, disp, mean 