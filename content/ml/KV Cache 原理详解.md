# KV Cache 原理详解

## 1. 为什么需要 KV Cache

自回归生成 (Autoregressive Generation) 的核心逻辑：

$$
P(x_{t+1} | x_1, x_2, \dots, x_t) = \text{Attention}(Q_t, K_{\leq t}, V_{\leq t})
$$

每次推理一个新 token，都需要重新计算**所有历史 token 的注意力**：

$$
O_t = \text{Attention}(Q_t, [K_1, \dots, K_t], [V_1, \dots, V_t])
$$

其中 $K_{\leq t}$ 和 $V_{\leq t}$ 包含大量**重复计算**——每次新 token 只追加一个 $K_t, V_t$，但 $K_1, \dots, K_{t-1}$ 被重新计算了 $t-1$ 次。

**KV Cache 的核心思想**：缓存已经计算过的 $K$ 和 $V$，避免重复计算。

## 2. KV Cache 的工作流程

### 预fill阶段 (Prefill)

```
输入: prompt tokens [x1, x2, ..., x_n]
输出: 第一个新 token x_{n+1}

1. 前向计算所有 prompt tokens 的 K, V
2. 缓存: K_cache = [K_1, ..., K_n], V_cache = [V_1, ..., V_n]
3. 计算第 n+1 步的 Q_n+1
4. Attention(Q_n+1, K_cache, V_cache) → 输出
5. 采样得到 x_{n+1}
```

### 生成阶段 (Generation / Decode)

```
新输入: x_{n+1}
1. 计算 x_{n+1} 的 K_{n+1}, V_{n+1}
2. 追加到缓存: K_cache.append(K_{n+1}), V_cache.append(V_{n+1})
3. Q 只取最后一个: Q = Q_n+1
4. Attention(Q, K_cache, V_cache) → 输出
5. 采样得到 x_{n+2}
... 重复 2-5
```

**关键**：每次只需计算**单个新 token** 的 $K, V$，而非重新计算全量 $K, V$。Attention 复杂度从 $O(t^2 d)$ 降至 $O(t d)$（单步）。

## 3. 为什么没有 Q Cache

### Q 的位置编码特性

$Q$ 携带的是**查询 token 本身的信息**，它与**该 token 在序列中的位置强绑定**。

```
位置 1 的 token 计算 Q_1，位置 2 的 token 计算 Q_2
当生成第 3 个 token 时，需要的 Q 是 "当前位置的查询"
即 Q_t 是当前时间步的函数，与历史 Q_t' (t' < t) 完全不同
```

### Attention 的方向不同

- **Q** 是"主动方"：当前 token 发出的查询
- **K, V** 是"被动方"：所有历史 token 提供的键值

Attention 的物理含义：

$$
O_t = \sum_{i=1}^{t} \text{softmax}(Q_t \cdot K_i^T) \cdot V_i
$$

$Q_t$ **不是**历史累积量，而是**当前时间步的单一查询**。它没有"历史累积"的需求，因为下一个 token 的 $Q$ 完全不同。

### 几何直觉

| 缓存对象 | 是否可缓存 | 原因 |
|---------|-----------|------|
| K | ✅ | 静态存储，历史 token 的键不随时间改变 |
| V | ✅ | 静态存储，历史 token 的值不随时间改变 |
| Q | ❌ | 动态生成，每个时间步的查询是当前位置的函数 |

## 4. KV Cache 的实现细节

### 显存占用

对于每个 token，存储两个向量 $K_t, V_t \in \mathbb{R}^d$：

$$
\text{显存} = 2 \times N_{\text{max}} \times d \times \text{bytes}(\text{precision})
$$

例如 batch=1, $N_{\text{max}}=4096$, $d=128$, FP16：

$$
2 \times 4096 \times 128 \times 2 \text{ bytes} \approx 2 \text{ MB}
$$

但实际上层数和头数使得真实占用大得多：

$$
2 \times N_{\text{max}} \times d \times \text{num\_layers} \times \text{num\_heads} \times \text{bytes}
$$

Llama 7B (40层, 32头, d=128) 处理 4096 序列长度，FP16 下 KV cache 约：

$$
2 \times 4096 \times 128 \times 40 \times 32 \times 2 \text{ bytes} \approx 256 \text{ MB}
$$

### 量化优化

为降低显存占用，常见方法：

- **FP8 KV Cache**：H100/H200 支持，显存减半
- **K-V Int8 量化**：如 LLM.int8()
- **分组量化 (Group-wise)**：按头或层分组量化

### Paged Attention (vLLM)

传统 KV Cache 的问题：显存碎片化（可变长度序列）。

vLLM 的 Paged Attention：类比操作系统的虚拟内存分页，将 KV Cache 划分为固定大小的块，实现显存的高效复用和多请求共享。

## 5. 常见问题

### Q 为什么不用 Cache？

**本质上 Q 是"当前位置的查询向量"，不是历史累积量**。每个生成步骤的 Q 来自当前 token 的 embedding 和位置编码，与之前所有 Q 正交等效。

### Cache 什么时候失效？

- **Prefix 复用**：相同 system prompt + user query 前缀，多个请求可共享 prompt 的 KV cache（如 ChatML 格式）
- **滑动窗口**：部分实现（如 Mistral）使用滑动窗口 Attention，旧的 KV 在窗口外被丢弃
- ** eviction 策略**：当 cache 满时，按 LRU 或 FIFO 驱逐

## 6. 参考

- vLLM: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Rolling Attention: [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14060)