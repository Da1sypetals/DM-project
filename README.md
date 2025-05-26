# DM

输入：统一为`(num_user, embedding_dim)`.

# 流程
将原始数据跑data_prep.py和sub_data.py，得到embedding和subreddit的数据；
送入similarity.py，得到recommendation；
将recommendation和subreddit数据送入eval.py，指定topk（要推荐topk个用户给每个用户），得到准确率。iou大于0都算准。

# 方法

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


# 输出格式
- 相似度：key是当前node，value是和当前node最匹配的topk nodes
  ```json
  "1": [
      {
        "node": 817,
        "sim": 0.9668042934535831
      },
      {
        "node": 715,
        "sim": 0.8588211330163609
      },
      {
        "node": 599,
        "sim": 0.8563563228457022
      },
      {
        "node": 165,
        "sim": 0.8538623626878953
      },
      {
        "node": 388,
        "sim": 0.8535572957188746
      }
    ],
  ```
- 聚类：node_id : cluster_id

## For my information
- 代码存在这里：`~/dev/dm`
