import os
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize,load_data
from img_deal import image_crop, extract_image_feat
from pathlib import Path
from calculate_adj import calculate_fadj,calculate_madj,calculate_sadj
from STESH_model import *
from STESH_train import train


def load_DLPFC_ST_file(dataset, highly_genes, k, radius, quality ='hires',image_path = None,library_id = None):
    path = "../data/DLPFC/" + dataset + "/"
    save_path = "../"
    labels_path = path + "metadata.tsv"
    print(labels_path)
    labels = pd.read_csv(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    
    NA_labels = np.where(labels.isnull()) 
    labels = labels.drop(labels.index[NA_labels[0]])  
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)

    
    adata1 = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()
    obs_names = np.array(adata1.obs.index) 
    positions = adata1.obsm['spatial'] 
    
    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)  # type: ignore
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0) # type: ignore
    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))
    
    adata.var_names_make_unique()
   
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.obs['array_row'] = adata1.obs['array_row']
    adata.obs['array_col'] = adata1.obs['array_col']
    adata.uns['spatial'] = adata1.uns['spatial']
    adata.var['gene_ids'] = adata1.var['gene_ids']
    adata.var['feature_types'] = adata1.var['feature_types']
    adata.var['genome'] = adata1.var['genome']

    adata.var_names_make_unique() 
    adata = normalize(adata, highly_genes=highly_genes)

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0) # type: ignore
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale

    adata.obs["imagecol"] = image_coor[:, 0] # type: ignore
    adata.obs["imagerow"] = image_coor[:, 1] # type: ignore
    adata.uns["spatial"][library_id]["use_quality"] = quality
    
    save_path_image_crop = Path(os.path.join(save_path, 'Image_crop', dataset))
    save_path_image_crop.mkdir(parents=True, exist_ok=True)
    adata = image_crop(adata, save_path=save_path_image_crop)
    adata = extract_image_feat(adata)
    adata.var_names_make_unique() 

    fadj = calculate_fadj(adata.X, k=k)
    sadj, graph_nei_sadj, graph_neg_sadj = calculate_sadj(adata, radius=radius)
    madj, graph_nei_madj, graph_neg_madj = calculate_madj(adata, k=k)

    adata.obsm["fadj"] = fadj 
    adata.obsm["sadj"] = sadj 
    adata.obsm["graph_nei_sadj"] = graph_nei_sadj.numpy() 
    adata.obsm["graph_neg_sadj"] = graph_neg_sadj.numpy() 
    adata.obsm["madj"] = madj
    adata.obsm["graph_nei_madj"] = graph_nei_madj.numpy() 
    adata.obsm["graph_neg_madj"] = graph_neg_madj.numpy() 
    adata.var_names_make_unique() 
    print("saved")
    return adata

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # datasets = ['151507', '151508', '151509', '151510', '151669', '151670',
    #             '151671', '151672', '151673','151674', '151675','151676']
    datasets = ['151673']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        savepath = "../output/" 
        adata = load_DLPFC_ST_file(dataset, highly_genes=3000, k=14, radius=560)
        adata.write('../output/'+ dataset +'_pre.h5ad')
        print("h5ad saved!")
        adata, features, labels, fadj, sadj, madj ,graph_nei_sadj, graph_neg_sadj , graph_nei_madj , graph_neg_madj = load_data(dataset)

        n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7
        
        adata, ARI, NMI, FMI, emb_max,idx_max,mean_max = train(dataset, adata,n_clusters,labels,features, sadj, fadj, madj , graph_nei_sadj, graph_neg_sadj,graph_nei_madj , graph_neg_madj)
        
        
        # plt.rcParams["figure.figsize"] = (3, 3)
        # # if not os.path.exists(savepath):
        # #     os.mkdir(savepath)
        # title = "Manual annotation (slice #" + dataset + ")"
        # sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title,show=False)
        # plt.savefig('../result/' + dataset + '_Manual Annotation.jpg', bbox_inches='tight', dpi=600)
        # plt.show() 
        
        
        title = 'STESH: ARI={:.2f}'.format(ARI) +' NMI={:.2f}'.format(NMI) +' FMI={:.2f}'.format(FMI)
        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig('../result/DLPFC/' + dataset + '_STESH.jpg', bbox_inches='tight', dpi=600) # type: ignore
        plt.show() # type: ignore
        
        
        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2,show=False)
        plt.savefig('../result/DLPFC/' + dataset + '_STESH_paga.jpg', bbox_inches='tight', dpi=600) # type: ignore
        plt.show() # type: ignore
        
        
        pd.DataFrame(emb_max).to_csv('../result/DLPFC/' + dataset + '_STESH_emb.csv')
        pd.DataFrame(idx_max).to_csv('../result/DLPFC/' + dataset + '_STESH_idx.csv')

        adata.layers['X'] = adata.X # type: ignore
        adata.layers['mean'] = mean_max # type: ignore
        adata.write('../result/DLPFC/' + dataset + '_STESH.h5ad') # type: ignore
        print(dataset + 'success!')
