# HimNet
Code for Graph-level Anomaly Detection via Hierarchical Memory Networks ([HimNet](https://arxiv.org/pdf/2307.00755.pdf)) (ECML-PKDD 2023)

## Abstract
Graph-level anomaly detection aims to identify abnormal graphs that exhibit deviant structures and node attributes compared to the majority in a graph set. One primary challenge is to learn normal patterns manifested in both fine-grained and holistic views of graphs for identifying graphs that are abnormal in part or in whole. To tackle this challenge, we propose a novel approach called Hierarchical Memory Networks (HimNet), which learns hierarchical memory modules – node and graph memory modules – via a graph autoencoder network architecture. The node-level memory module is trained to model fine-grained, internal graph interactions among nodes for detecting locally abnormal graphs, while the graph-level memory module is dedicated to the learning of holistic normal patterns for detecting globally abnormal graphs. The two modules are jointly optimized to detect both locally- and globally anomalous graphs.
![Framework](framework.PNG)

## Data Preparation

Some of datasets are in ./dataset folder. Due to the large file size limitation, some datasets are not uploaded in this project. You may download them from the urls listed in the paper.

## Train
For datasets except HSE, p53, MMP, PPAR-gamma and hERG, run the following code:
    
    python train.py --DS dataset --feature default/deg-num

For HSE, p53, MMP and PPAR-gamma, run the following code:

    python train_tox.py --DS dataset

For hERG, run the following code:

    python train_herg.py


## Citation
```bibtex
@inproceedings{niu2023graph,
  title={Graph-Level Anomaly Detection via Hierarchical Memory Networks},
  author={Niu, Chaoxi and Pang, Guansong and Chen, Ling},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={201--218},
  year={2023},
  organization={Springer}
}
```
