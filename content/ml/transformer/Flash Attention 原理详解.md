# Flash Attention 原理详解

## 1. 背景：标准 Attention 的问题

标准 Self-Attention 计算：

$$
S = QK^T \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S), \quad O = PS
$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 为序列长度，$d$ 为维度。

**问题**：需要 Materialize 完整注意力矩阵 $S$ 和 $P$，分别需要 $O(N^2)$ HBM 访问和显存。

## 2. Flash Attention v1 (2022) - IO-Aware Attention

### 核心思想

不一次性 materialization 完整矩阵，利用分块计算 (Tiling) 减少 HBM (High Bandwidth Memory) 访问。

### 算法流程

```
输入: Q, K, V (分块 T x B)
输出: O

1. 将 Q, K, V 分块
2. 初始化: Oi = zeros, li = zeros, mi = -inf
3. 对每个块 j:
   - 从 HBM 加载 Kj, Vj
   - 计算 Si = Q_i @ Kj^T
   - 对 Si 行归一化: pi = softmax(mi + Si)  # 需额外向量 m_i 记录 max
   - 更新: li_new = sum(pi, axis=1)
   - Oi_new = diag(li_new)^(-1) @ (diag(li) @ Oi + pi @ Vj)
   - mi_new = max(mi, max(Si, axis=1))
   - 更新 li, mi, Oi
```

### 关键优化

- **分块矩阵乘法** (Tiled Matrix Multiplication)
- **在线 softmax** (Online Softmax)：用 $m_i$ 和 $l_i$ 递推，避免存储完整 softmax 结果
- **重计算** (Recomputation)：反向时不存储注意力矩阵，而是重新计算

### 复杂度

- 计算量：$O(N^2 \cdot d)$ (不变)
- HBM 访问：$O(N^2)$ → $O(Nd)$ per block
- 显存：$O(N^2)$ → $O(N)$

## 3. Flash Attention v2 (2023) - 进一步优化

### 主要改进

1. **Sequence Parallel**：将序列维度切分，支持多卡并行
2. **减少非矩阵乘法 (Non-MatMul) 运算**：对因果注意力 (Causal Attention) 优化
3. **更好的分块策略**：调整块大小以提高 GPU 利用率
4. **支持 Variable Length**：处理不同长度的序列

### Sequence Parallel (序列维度并行)

#### 为什么需要 SP？

标准 Attention 计算中，序列长度 $N$ 通常远大于 batch size。对于超长序列（如 $N=32768$），即使单卡 A100 80GB 也难以容纳完整注意力矩阵。

传统数据并行 (Data Parallel) 在 batch 维度切分，但 batch 通常只有 1，无法有效利用多卡。

#### SP 核心思路

将序列维度 $N$ 在多卡间切分，每卡持有 $N/P$ 长度的 $Q_i, K_i, V_i$。

#### Attention 的并行化困难

Attention 计算具有**全局依赖性**：输出 $O_i$ 理论上需要所有 $K, V$ 的信息。

$$
O_i = \sum_{j=1}^{N} \text{softmax}(Q_i \cdot K_j^T) \cdot V_j
$$

这意味着即使把 $Q$ 切分了，每个输出 token 仍然需要汇聚**全局的** $K$ 和 $V$。

#### Flash Attention v2 的 SP 实现

**关键洞察**：利用在线 softmax 的递推公式，将聚合操作延迟到最后。

回忆在线 softmax 维护的两个统计量：

$$
m_i = \max_j S_{ij}, \quad l_i = \sum_j e^{S_{ij} - m_i}
$$

输出为：

$$
O_i = \frac{1}{l_i} \sum_j e^{S_{ij} - m_i} V_j
$$

**分步聚合协议**（2D 切分，每卡持有 $Q_i$）：

```
假设有 P 张卡，每卡处理序列段 Qi (长度 N/P)
每张卡需要访问全局的 K 和 V

1. All-Gather: 各卡将自己的 Ki, Vi 广播给其他卡
2. 本地计算: 每张卡计算 local Si = Qi @ (Kj 的部分)^T
3. Reduce-Scatter: 使用在线 softmax 的增量公式合并不同卡的结果
4. 最终输出: 每张卡持有 Oi 的对应片段
```

**更高效的实现——环状通信 (Ring Attention)**：

```
               ┌─────────────────────────────────────────┐
               │  Ring: Q1 → Q2 → Q3 → Q4 → Q1          │
               │  每一步:                               │
               │    1. 接收上游的 Kj, Vj                │
               │    2. 计算 Si = Qi_local @ Kj^T        │
               │    3. 用增量公式更新 mi, li, Oi        │
               │    4. 转发给下游                        │
               │  P 步后，每张卡持有完整 Oi             │
               └─────────────────────────────────────────┘
```

每个 token 的输出需要遍历所有 $P$ 个 $K/V$ 分块，总通信量 $O(Nd \cdot P)$，通信步数与卡数成正比，适合高带宽互联 (NVLink)。

**与张量并行的对比**：

| 维度 | Tensor Parallel (TP) | Sequence Parallel (SP) |
|------|---------------------|------------------------|
| 切分维度 | 隐藏维度 $d$ | 序列维度 $N$ |
| 通信模式 | All-Gather + Reduce-Scatter (每层) | Ring Pass ($P$ 步/层) |
| 通信量 | $O(Nd)$ per layer | $O(Nd)$ per layer |
| 适用场景 | 中等序列 | 超长序列 |

### 与 v1 核心差异

```
Flash Attention v1:
- 主要优化计算流程
- 有限的并行度

Flash Attention v2:
- 新增 sequence parallel (SP) 支持
- 优化了 causal mask 场景
- 分块大小从 64/128 调整为更灵活的设置
```

## 4. Flash Attention v3 (2024) - Hopper 架构优化

### 新特性

1. **Tensor Memory Controller (TMA)**：专用硬件支持，更高效的内存访问
2. **Asynchronous Data Loading**：异步数据加载与计算重叠
3. **Warp Specialization**：不同 warp group 执行不同任务
4. **FP8 支持**：低精度计算加速

### 核心算法变化

```
- 使用 TMA 指令进行更高效的 HBM 访问
- Warp specialization: 部分线程专门加载数据，另一部分计算
- 异步执行隐藏内存访问延迟
```

## 5. 各版本对比

| 特性 | v1 | v2 | v3 |
|------|-----|-----|-----|
| 年份 | 2022 | 2023 | 2024 |
| 核心优化 | 分块+在线softmax | SP+Non-MatMul优化 | TMA+异步+WarpSpec |
| 并行 | 数据并行 | Sequence Parallel | 多层次并行 |
| 精度 | FP16/BF16 | FP16/BF16 | FP16/BF16/FP8 |
| 主要硬件 | A100 | A100/H100 | H100 |

## 6. 关键公式推导

### 在线 Softmax

标准 softmax：
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

在线版本引入：
$$
m_i = \max_j x_j, \quad l_i = \sum_j e^{x_j - m_i}
$$
$$
\text{softmax}(x_i) = \frac{e^{x_i - m_i}}{l_i}
$$

增量更新时：
$$
m^{(t)} = \max(m^{(t-1)}, \max_j x_j^{(t)})
$$
$$
l^{(t)} = l^{(t-1)} \cdot e^{m^{(t-1)} - m^{(t)}} + \sum_j e^{x_j^{(t)} - m^{(t)}}
$$

## 7. 使用示例

```python
# PyTorch >= 2.0
import torch
from torch.nn.functional import scaled_dot_product_attention

# 自动使用 Flash Attention
output = scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True
)

# HuggingFace Transformers
# 设置 use_flash_attention=True
```

## 8. 参考

- Flash Attention v1: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Flash Attention v2: [FlashAttention-2: Faster Attention with Better Parallelism and Worker Partitioning](https://arxiv.org/abs/2307.08691)
- Flash Attention v3: [FlashAttention-3: Fast and Accurate Attention with FP8 and Better Parallelism](https://arxiv.org/abs/2404.09526)