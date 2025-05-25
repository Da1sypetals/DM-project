#!/usr/bin/env python3
"""
基于用户 embedding 构建 kNN 加权图

功能:
  - 从 .npy 文件加载用户 embedding (n_users, emb_dim)
  - 计算两两余弦相似度
  - 对每个节点选择 k 个最近邻 (最大相似度)
  - 构建 NetworkX 无向图，边带权重 (相似度)
  - 打印图的基本信息和部分边示例

用法:
  python build_graph.py --emb embeddings.npy --k 5
"""

import argparse
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(path: str) -> np.ndarray:
    """加载 embedding .npy 文件"""
    return np.load(path)


def build_knn_graph(emb: np.ndarray, k: int) -> nx.Graph:
    """
    构建 kNN 图:
      - 使用余弦相似度计算全图相似度矩阵
      - 对每个节点, 连接到相似度最高的 k 个节点
    返回:
      - NetworkX Graph, 节点为 0..n_users-1, 边权重为相似度
    """
    n_users = emb.shape[0]
    # 计算余弦相似度
    sim_matrix = cosine_similarity(emb)
    # 清除自环
    np.fill_diagonal(sim_matrix, 0)

    G = nx.Graph()
    G.add_nodes_from(range(n_users))

    # 对每个节点，添加 k 边
    for i in range(n_users):
        # 找到 top-k 相似度节点索引
        neighbors = np.argpartition(-sim_matrix[i], k)[:k]
        for j in neighbors:
            weight = float(sim_matrix[i, j])
            # 添加无向边，若已存在保留最大权重
            if G.has_edge(i, j):
                if G[i][j]["weight"] < weight:
                    G[i][j]["weight"] = weight
            else:
                G.add_edge(i, j, weight=weight)
    return G


def main():
    parser = argparse.ArgumentParser(description="构建 kNN 加权图")
    parser.add_argument("--emb", required=True, help="输入 embedding .npy 路径")
    parser.add_argument("-k", type=int, default=5, help="每个节点的近邻数 k")
    args = parser.parse_args()

    emb = load_embeddings(args.emb)
    G = build_knn_graph(emb, args.k)

    # 打印图信息
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # 打印前 10 条边示例
    print("Sample edges (node_i, node_j, weight):")
    count = 0
    for u, v, attr in G.edges(data=True):
        print(f"{u}\t{v}\t{attr['weight']:.4f}")
        count += 1
        if count >= 10:
            break


if __name__ == "__main__":
    main()
