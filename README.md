# HimNet
Code for Graph-level Anomaly Detection via Hierarchical Memory Networks (HimNet) (ECML-PKDD 2023)

## Abstract
Graph-level anomaly detection aims to identify abnormal graphs that exhibit deviant structures and node attributes compared to the majority in a graph set. One primary challenge is to learn normal patterns manifested in both fine-grained and holistic views of graphs for identifying graphs that are abnormal in part or in whole. To tackle this challenge, we propose a novel approach called Hierarchical Memory Networks (HimNet), which learns hierarchical memory modules – node and graph memory modules – via a graph autoencoder network architecture. The node-level memory module is trained to model fine-grained, internal graph interactions among nodes for detecting locally abnormal graphs, while the graph-level memory module is dedicated to the learning of holistic normal patterns for detecting globally abnormal graphs. The two modules are jointly optimized to detect both locally- and globally anomalous graphs.
![Framework](framework.PNG)

## Data Preparation

Some of datasets are in ./dataset folder. Due to the large file size limitation, some datasets are not uploaded in this project. You may download them from the urls listed in the paper.

## Train



## Citation
```bibtex
@inproceedings{niuhimnet,
  title={Graph-level Anomaly Detection via Hierarchical Memory Networks},
  author={Niu, Chaoxi and Pang, Guansong and Chen, Ling},
  booktitle=" European Conference on Machine Learning and Knowledge Discovery in Databases",
  year={2023}
}
```
