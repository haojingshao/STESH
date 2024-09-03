import os
import random
import numpy as np
import pandas as pd 
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torchvision.transforms as transforms

def image_crop(adata,save_path,library_id=None,crop_size=50,target_size=224,verbose=False):
    if library_id is None:
         library_id = list(adata.uns["spatial"].keys())[0]
        
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    image = adata.uns["spatial"][library_id]["images"][
         adata.uns["spatial"][library_id]["use_quality"]]
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
        
    img_pillow = Image.fromarray(image)
    tile_names = []
    
    with tqdm(total=len(adata),
              desc="Tiling image",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
                
            tile.thumbnail((target_size, target_size),Image.Resampling.LANCZOS)
            tile.resize((target_size, target_size)) 
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)
    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata



def load_cnn_model(cnnType,device):
    if cnnType == 'ResNet50':
        cnn_pretrained_model = models.resnet50(pretrained=True)
        cnn_pretrained_model.to(device)
    elif cnnType == 'Resnet152':
        cnn_pretrained_model = models.resnet152(pretrained=True)
        cnn_pretrained_model.to(device)
    elif cnnType == 'Vgg19':
        cnn_pretrained_model = models.vgg19(pretrained=True)
        cnn_pretrained_model.to(device)
    elif cnnType == 'Vgg16':
        cnn_pretrained_model = models.vgg16(pretrained=True)
        cnn_pretrained_model.to(device)
    elif cnnType == 'DenseNet121':
        cnn_pretrained_model = models.densenet121(pretrained=True)
        cnn_pretrained_model.to(device)
    elif cnnType == 'Inception_v3':
        cnn_pretrained_model = models.inception_v3(pretrained=True)
        cnn_pretrained_model.to(device)
    else:
        raise ValueError(f"{cnnType} is not a valid type.")
    return cnn_pretrained_model
    

def extract_image_feat(adata, pca_components=50, cnnType='ResNet50', device=False, verbose=False, seeds=88):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feat_df = pd.DataFrame()
    cnn_model = load_cnn_model(cnnType,device)
    cnn_model.eval()

    if "slices_path" not in adata.obs.keys():
        raise ValueError("Please run the function image_crop first")
        
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
                      transforms.RandomAutocontrast(),
                      transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                      transforms.RandomInvert(),
                      transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                      transforms.RandomSolarize(random.uniform(0, 1)),
                      transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                      transforms.RandomErasing()
                      ]
    
    with tqdm(
        total=len(adata),
        desc="Extract image feature",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for spot, slice_path in adata.obs['slices_path'].items():
            spot_slice = Image.open(slice_path) 
            spot_slice = spot_slice.resize((224, 224))  
            spot_slice = np.asarray(spot_slice, dtype="int32")  
            spot_slice = spot_slice.astype(np.float32) 
            tensor = transforms.Compose(transform_list)(spot_slice) 
            tensor = tensor.resize_(1, 3, 224, 224)   # type: ignore 
            tensor = tensor.to(device)
            result = cnn_model(Variable(tensor)) 
            result_npy = result.data.cpu().numpy().ravel()
            feat_df[spot] = result_npy 
            feat_df = feat_df.copy() 
            pbar.update(1) 
    adata.obsm["image_feat"] = feat_df.transpose().to_numpy() 
    if verbose:
        print("The image feature is added to adata.obsm['image_feat'] !")
        
    pca = PCA(n_components=pca_components, random_state=seeds) 
    pca.fit(feat_df.transpose().to_numpy())
    adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy()) 
    if verbose:
        print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
    return adata