import torch
import torch.nn as nn
import torch.nn.functional as F


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1), 
            torch.nn.BatchNorm1d(nhid1), 
            torch.nn.ReLU() 
        )
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6) 

    def forward(self, emb):
        x = self.decoder(emb) 
        disp = self.DispAct(self.disp(x)) 
        mean = self.MeanAct(self.mean(x)) 
        return [disp, mean]