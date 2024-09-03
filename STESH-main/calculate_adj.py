import torch
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import numpy as np
import pandas as pd
import scipy.sparse as sp


def calculate_fadj(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    
    print("start calculate_fadj!")
    if pca is not None:
        features = PCA(n_components=50).fit_transform(features).reshape(-1, 1) 
    print("The number of nearest neighbors ", k)
    
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True) # type: ignore 
    A = A.toarray() 
    row, col = np.diag_indices_from(A) 
    A[row, col] = 0 
    fadj = sp.coo_matrix(A, dtype=np.float32) 
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj) 
    return fadj


def calculate_sadj(adata, radius=150):
    
    print("start calculate_sadj!")
    coor = pd.DataFrame(adata.obsm['spatial']) 
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0])) 
    
    print("The nearest neighbor radius: ",radius)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor) # type: ignore 
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True) 

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]]=1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs)) # type: ignore
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs)) # type: ignore

    graph_nei_sadj = torch.from_numpy(A) 
    graph_neg_sadj = torch.ones(coor.shape[0],coor.shape[0]) - graph_nei_sadj

    sadj = sp.coo_matrix(A, dtype=np.float32) 
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj) 
    return sadj, graph_nei_sadj,  graph_neg_sadj 

def calculate_madj(adata, k=15, mode="connectivity", metric="cosine"):
    
    print("start calculate_madj!")
    coor = pd.DataFrame(adata.obsm['image_feat_pca']) 
    
    correlation_matrix = np.corrcoef(adata.obsm['image_feat_pca']) 
    A = np.zeros((coor.shape[0], coor.shape[0])) 
    
    print("The number of nearest neighbors ", k)
    for i in range(coor.shape[0]):
        distances = correlation_matrix[i]
        nearest_neighbors = np.argsort(distances)[::-1][1:k+1]  
        A[i, nearest_neighbors] = 1
        A[nearest_neighbors, i] = 1  

    print('The graph contains %d edges, %d cells.' % (np.sum(A), coor.shape[0]))
    print('%.4f neighbors per cell on average.' % (np.sum(A) / coor.shape[0])) # type: ignore

    graph_nei_madj = torch.from_numpy(A) 
    graph_neg_madj = torch.ones(coor.shape[0], coor.shape[0]) - graph_nei_madj 

    madj = sp.coo_matrix(A, dtype=np.float32) 
    madj = madj + madj.T.multiply(madj.T > madj) - madj.multiply(madj.T > madj) 

    return madj,  graph_nei_madj ,  graph_neg_madj 