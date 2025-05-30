# DEC output 含义解释

1. **`latent.npy` → 潜在表示 Z**

   * 形状：$(n\_users,\; latent\_dim)$
   * 含义：第 $i$ 行向量 $z_i$ 是自编码器（及后续迭代微调）学到的“潜在空间”表示。它比原始 $(emb\_dim)$ 维度更低，且在训练过程中被优化得更有利于区分不同簇（论坛社区），去除了噪声与冗余维度。
   * 用途：可直接用于可视化（如 t-SNE）、下游聚类或相似度计算，也可以作为后续任务的特征输入。

2. **`q.npy` → 软分配概率矩阵 Q**

   * 形状：$(n\_users,\; n\_clusters)$
   * 含义：第 $i$ 行 $\mathbf{q}_i = [q_{i1}, q_{i2}, \dots, q_{iK}]$ 表示用户 $i$ 落入每个簇（论坛）$c$ 的“置信概率”：

     $$
       q_{ic}
       \;=\;
       \frac{(1 + \|z_i - \mu_c\|^2 / \alpha)^{-\frac{\alpha+1}{2}}}
            {\sum_{c'} (1 + \|z_i - \mu_{c'}\|^2 / \alpha)^{-\frac{\alpha+1}{2}}}
     $$

     在我们实现中取 $\alpha=1$。值越大说明 $i$ 越“偏好”或更“靠近”簇中心 $\mu_c$。每行和为 1。
   * 用途：

     * 可以直接用来衡量一个用户对各论坛归属的模糊度；
     * 在无监督场景下，提供比硬分配更丰富的信息；
     * 可用于后续构造两两同论坛的联合概率（$\sum_c q_{ic}q_{jc}$）。

3. **`y_pred.npy` → 硬聚类标签**

   * 形状：$(n\_users,)$
   * 含义：第 $i$ 个元素是用户 $i$ 最终被分配到的簇（论坛）编号：

     $$
       y_i = \arg\max_{c} \; q_{ic}.
     $$

     取软分配概率最大的那个簇作为硬标签。
   * 用途：

     * 为每个用户提供一个清晰的“社群归属”；
     * 方便统计各簇规模、对簇内用户做后续分析（如社区行为、内容偏好等）。

