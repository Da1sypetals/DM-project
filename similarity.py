#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from typing import List, Tuple, Dict


def load_embeddings(path: str) -> np.ndarray:
    """
    从 .npy 文件加载 embedding。
    """
    return np.load(path)


def compute_user_similarities(
    embeddings: np.ndarray,
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    使用 NumPy 向量化计算所有用户对的欧氏、余弦和 Jaccard 相似度，中间不排序。

    返回 dict，键为度量名称，值为未排序的 (i, j, sim) 列表。
    """
    n_users = embeddings.shape[0]

    # Jaccard 二值化
    eps = 1e-8
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    normed = (embeddings - mean) / (std + eps)
    bin_emb = (normed > 0).astype(int)

    # 上三角对
    iu, ju = np.triu_indices(n_users, k=1)

    # 欧氏相似度
    diff = embeddings[iu] - embeddings[ju]
    eu_sim = 1.0 / (1.0 + np.linalg.norm(diff, axis=1))

    # 余弦相似度
    norms = np.linalg.norm(embeddings, axis=1)
    dots = (embeddings @ embeddings.T)[iu, ju]
    denom = (norms[iu] * norms[ju]) + eps
    cos_sim = dots / denom

    # Jaccard 相似度
    inter = (bin_emb @ bin_emb.T)[iu, ju].astype(float)
    unions = (bin_emb.sum(axis=1)[iu] + bin_emb.sum(axis=1)[ju] - inter) + eps
    jac_sim = inter / unions

    return {
        "euclidean": list(zip(iu.tolist(), ju.tolist(), eu_sim.tolist())),
        "cosine": list(zip(iu.tolist(), ju.tolist(), cos_sim.tolist())),
        "jaccard": list(zip(iu.tolist(), ju.tolist(), jac_sim.tolist())),
    }


def sort_and_topk(
    sims: List[Tuple[int, int, float]], topk: int
) -> List[Tuple[int, int, float]]:
    """
    对相似度列表降序排序并截取前 topk 项。
    如果 topk=None，则返回全部排序结果。
    """
    sorted_pairs = sorted(sims, key=lambda x: x[2], reverse=True)
    return sorted_pairs if topk is None else sorted_pairs[:topk]


def main():
    parser = argparse.ArgumentParser(description="用户嵌入相似度计算 CLI")
    parser.add_argument(
        "--emb", type=str, required=True, help="输入 embedding .npy 文件路径"
    )
    parser.add_argument("-o", "--output", type=str, help="输出文件路径，默认 stdout")
    parser.add_argument(
        "--topk", type=int, default=None, help="每种度量输出的前 topk 项，默认全部"
    )
    args = parser.parse_args()

    emb = load_embeddings(args.emb)
    sims = compute_user_similarities(emb)

    # 准备输出字符串
    out_lines = []
    for metric, pairs in sims.items():
        top_pairs = sort_and_topk(pairs, args.topk)
        out_lines.append(f"=== {metric.upper()} 前{args.topk or '全部'} 相似度 ===")
        for i, j, sim in top_pairs:
            out_lines.append(f"{i}\t{j}\t{sim:.6f}")
        out_lines.append("")

    output_text = "\n".join(out_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
