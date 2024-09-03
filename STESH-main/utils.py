import torch
import scanpy as sc
import numpy as np
import scipy.sparse as sp

def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1)) # type: ignore
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def normalize(adata, highly_genes=3000): 
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)  
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes) 
    adata = adata[:, adata.var['highly_variable']].copy() 
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000 
    sc.pp.scale(adata, zero_center=False, max_value=10) 
    return adata

def normalize_sparse_matrix(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape) # type: ignore


def mclust_R(adata, n_clusters, modelNames='EEE', use_rep='emb', key_added='STESH', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import os
    os.environ['R_HOME'] = '/data/changjiaxing/software/miniconda3/envs/STESH_env/lib/R'

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust") # type: ignore

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed) # type: ignore
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames) # type: ignore
    mclust_res = np.array(res[-2])
    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata
