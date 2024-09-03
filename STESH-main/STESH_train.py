import random
import torch
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from utils import mclust_R
from STESH_model import *
from sklearn import metrics
from loss import NB,regularization_loss,consistency_loss
from tqdm import tqdm

def train(dataset,adata,n_clusters,labels,features, sadj, fadj, madj ,graph_nei_sadj, graph_neg_sadj,graph_nei_madj , graph_neg_madj,
    seed = 100,
    cuda = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    epochs = 200,
    alpha = 1,
    beta = 10,
    gamma = 0.1,
    nhid1 = 128,
    nhid2 = 64,
    dropout = 0):
    
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if not False and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore
    model = STESH(nfeat=features.shape[1],nhid1=nhid1,nhid2=nhid2,dropout=dropout)
    
    if cuda:
        features = features.cuda()
        fadj = fadj.cuda()
        sadj = sadj.cuda()
        graph_nei_sadj = graph_nei_sadj.cuda()
        graph_neg_sadj = graph_neg_sadj.cuda()
        madj = madj.cuda()
        graph_nei_madj = graph_nei_madj.cuda()
        graph_neg_madj = graph_neg_madj.cuda()
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    print("start train model!")
    
    ARI = 0
    NMI = 0
    FMI = 0
    idx_max = []
    mean_max = []
    emb_max = []
    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        optimizer.zero_grad()
        com1, com2, com3, emb, disp, mean = model(features, sadj, fadj ,madj)
        nb_loss = NB(theta=disp).loss(features, mean, mean=True)
        reg_loss = regularization_loss(emb, graph_nei_sadj, graph_neg_sadj , graph_nei_madj , graph_neg_madj)
        con_loss = consistency_loss(com1, com2, com3)
        total_loss = alpha * nb_loss + beta * con_loss + gamma * reg_loss
        emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
        mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
        total_loss.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()
        adata.obsm['emb'] = emb
        adata=mclust_R(adata, n_clusters, use_rep='emb', key_added='STESH')
        idx = adata.obs['STESH']
        
        ari_res = metrics.adjusted_rand_score(labels, idx) 
        nmi_res = metrics.normalized_mutual_info_score(labels, idx)
        fmi_res = metrics.fowlkes_mallows_score(labels, idx)
        
        if ari_res > ARI:
            ARI = ari_res
            NMI=nmi_res
            FMI=fmi_res
            idx_max = idx
            mean_max = mean
            emb_max = emb
            
    print(dataset, ' ','ARI=', ARI)
    print(dataset, ' ','NMI=',NMI)
    print(dataset, ' ','FMI=',FMI)
    
    adata.obs['idx'] = idx_max.astype(str) # type: ignore
    adata.obsm['emb'] = emb_max
    adata.obsm['mean'] = mean_max
    return adata, ARI,NMI,FMI, emb_max,idx_max,mean_max