# 2021_Summer_Internship

## Laborartory

Data Science & Artificial Intelligence Laboratory ([DSAIL](http://dsail.kaist.ac.kr/))


## Research ideas

#### Experiments 1 & 2 on Deep Graph InfoMax
```
  Experiments 1. Corrupt function
  Experiments 2. Visualization by epoch (PCA, T-SNE)
```
- [Go to code](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/Graph%20Neural%20Network/DGI)

#### Data Augmentation Experiments

- [Augmenatation on DGI](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/Graph%20Neural%20Network/DGI_augment)

- [Augmentation on GCN](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/Graph%20Neural%20Network/GCN_augment)

## Seminar
### Basic recommender system and Graph Neural Network
| Year | Paper | About |
| :---: | :------------: |  :---: |
| 07/05| [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)|[GCN](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/Graph%20Neural%20Network/GCN)
| 07/05|[Graph Attention Network](https://arxiv.org/abs/1710.10903) |GAT
| 07/05| [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)|[Deepwalk](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/Graph%20Neural%20Network/RandomWalk/Deepwalk)
| 07/12|[Node2vec : Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) |node2vec
| 07/12| [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)|[Netflix](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/Netflix)
| 07/12| [Probabilistic Matrix Factorization](https://papers.nips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)|[PMF](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/PMF)
| 07/19| [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)|[OCCF](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/OCCF)
| 07/19|[BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)| BPR |
| 07/19| [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)| [FM](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/FM) |
| 07/26| [LINE : Large-scale Information Network Embedding](https://arxiv.org/abs/1503.03578)| [LINE](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/Graph%20Neural%20Network/RandomWalk/LINE) |
| 07/26 | [metapath2vec : Scalable Representation Learning for Heterogeneous Networks](https://dl.acm.org/doi/10.1145/3097983.3098036)| Metapath2Vec
| 07/26 |[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)| WD |
| 08/09 |[Deep Graph Infomax](https://arxiv.org/abs/1809.10341)| [DGI](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/Graph%20Neural%20Network/DGI) |
| 08/09 | [Inductive Representation Learning on Large Graphs](https://papers.nips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)| GraphSAGE |
| 08/17 | [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)| [TransE](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/Graph%20Neural%20Network/KnowledgeGraph/TransE) |


## Others

| Date | About|
|  :---:|  :---: |
| 07/11 | [SVD_basic.ipynb](https://github.com/rlagywns0213/2021_Summer_Internship/blob/main/RecSys/SVD_basic.ipynb)
| 07/11 | [User-based CF.ipynb](https://github.com/rlagywns0213/2021_Summer_Internship/blob/main/RecSys/User-based%20CF.ipynb)

---

### Compare Learning Algorithm on Explicit Datasets

**In experiment, You can see the trade-off relationship between the loss and the number of factors in ALS Algorithm.**

**Also, You can see that SGD outperforms ALS in terms of loss as the number of factors increases.**
  1. [Go to SGD code](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/SGD)

  2. [Go to ALS code](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/ALS)

---

### Compare adding bias on SVD model (Experiment in 50, 100, 200 factors)

  **In experiment, You can see the the improvement of the test loss in SVD model by adding bias.**

  - [Go to code](https://github.com/rlagywns0213/2021_Summer_Internship/tree/main/RecSys/Netflix)
