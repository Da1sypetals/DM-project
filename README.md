# DM

输入：统一为`(num_user, embedding_dim)`.

## 简单相似度聚类方法

直接对比embedding的距离并排序（cosine, euclidean, 经过二值化处理的 jaccard）
- similarity.py

## feature 进一步提取后的方法：

- VAE重建后用mu representation作为feature
  - model.py
  - train.py
  - transform_emb.py
- DEC https://arxiv.org/pdf/1511.06335
  - dec.py

## 基于图的方法：

graph_cluster.py

- 建图：根据embedding，KNN 建图
- Louvain
- Infomap
- LabelProp（效果很差）