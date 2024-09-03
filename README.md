# STESH: A Spatial Domain Recognition Method for Integrating Spatial Transcriptome Multimodal Information Based on Graph Deep Learning

## Overview

Identifying spatial domains is the first important step in spatial transcriptomics (ST). Histological information can provide insights beyond gene expression profiles. To make the most of this information, we propose **STESH**, a spatial transcriptomic clustering approach that combines gene expression, spatial information, and histology. STESH uses graph convolutional neural networks to extract histological features and generate expression, histological, spatial, and collaborative convolution modules for multi-view graph convolutional networks with decoders and attention mechanisms. The test results show that STESH outperforms other algorithms in most cases.

![Frame diagram](https://github.com/haojingshao/STESH/blob/main/STESH.png?raw=true)

## Tutorial

### 1. Start by grabbing the source code

```bash
git clone https://github.com/haojingshao/STESH.git
cd STESH
```
### 2. Create a virtual environment
```bash
conda create -n STESH_env python=3.9
conda activate STESH_env
```
#### Requirements

To run STESH, you need the following dependencies:

- Python==3.9.17
- R==4.2.0
- scanpy==1.9.4
- numpy==1.23.4
- pandas==2.1.0
- matplotlib==3.7.2
- scikit-learn==1.3.0
- scipy==1.9.1
- anndata==0.9.2
- Pillow==10.0.0
- opencv-python==4.8.0.76
- python-louvain==0.16
- rpy2==3.5.14
- torch==1.9.1+cu111
- torchvision==0.10.1+cu111
- torch_geometric==2.4.0
- torch-sparse==0.6.12
- torch-scatter==2.0.9
- tqdm==4.66.1

#### To use STESH in a Jupyter notebook, run:
```bash
pip install ipykernel
python -m ipykernel install --user --name=STESH_env
```

### 3. Example
We take 10X sample 151673 as a running example.
The tutorial can be found in STESH-main/DLPFC_tutorial.py.
The results can be viewed in the DLPFC folder under the result folder.

## Citation
补充
