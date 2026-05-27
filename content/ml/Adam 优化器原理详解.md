# Adam 优化器原理详解

## 1. 背景：为什么需要自适应优化器

随机梯度下降 (SGD) 的问题：

- 所有参数使用统一学习率 $\eta$
- 稀疏特征得不到充分更新
- 容易陷入局部最优或震荡

**核心矛盾**：不同参数梯度方差差异大，但共享同一学习率。

## 2. Adam 算法

Adam (Adaptive Moment Estimation) 结合了 Momentum 和 RMSProp 的优点。

### 算法流程

```
输入: 学习率 η, 衰减系数 β1, β2, 数值稳定项 ε, 梯度 g
输出: 参数更新 Δθ

1. 一阶矩估计 (动量):
   m_t = β1 · m_{t-1} + (1 - β1) · g_t

2. 二阶矩估计 (自适应学习率):
   v_t = β2 · v_{t-1} + (1 - β2) · g_t²

3. 偏置校正 (Bias Correction):
   m̂_t = m_t / (1 - β1^t)
   v̂_t = v_t / (1 - β2^t)

4. 参数更新:
   θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

### 默认超参数

- $\eta = 0.001$
- $\beta_1 = 0.9$ (一阶矩衰减)
- $\beta_2 = 0.999$ (二阶矩衰减)
- $\varepsilon = 10^{-8}$

### 直观理解

| 组件 | 作用 | 类比 |
|------|------|------|
| $m_t$ (动量) | 累积历史梯度方向，减少震荡 | 物理中的惯性 |
| $v_t$ (二阶矩) | 自适应学习率：梯度大的参数降低学习率 | 自适应步长 |
| 偏置校正 | 初始化时 $m_0=v_0=0$ 导致估计偏零 | 无偏估计 |

## 3. 数学推导

### 为什么需要偏置校正？

设 $g_i$ 是独立同分布噪声，$E[g_i] = 0$，$E[g_i^2] = \sigma^2$。

**二阶矩的期望**：$E[v_t] = (1-\beta_2^t) \sigma^2 + O(\beta_2^t)$

当 $t$ 很小时，$1-\beta_2^t$ 接近 0，导致 $v_t$ 被低估。

校正后：

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t} \approx \sigma^2 + O(\beta_2^t \sigma^2) = E[g_t^2]
$$

这使得早期步骤的估计更准确。

### 收敛性

Adam 的 regret bound：

$$
R(T) \leq \frac{\| \theta_1 - \theta^* \|^2}{2\eta(1-\beta_1)} + \frac{\eta \sqrt{1-\beta_2}}{2(1-\beta_1)^{3/2}} \sum_{t=1}^T \| g_{1:t} \|_2
$$

但实践中发现，Adam 在某些任务（如 fine-tuning）不如 SGD + Momentum 稳定。

## 4. Adam vs 其他优化器

| 优化器 | 更新规则 | 特点 |
|--------|---------|------|
| SGD | $\theta \leftarrow \theta - \eta g$ | 简单、稳定，但收敛慢 |
| SGD + Momentum | $\theta \leftarrow \theta - \eta \cdot m_t$ | 加速收敛，减少震荡 |
| AdaGrad | $\theta \leftarrow \theta - \eta / \sqrt{G_t + \epsilon}$ | 适合稀疏梯度， learning rate 单调递减 |
| RMSProp | $\theta \leftarrow \theta - \eta \cdot g / \sqrt{v_t}$ | 指数移动平均，非稀疏友好 |
| Adam | $\theta \leftarrow \theta - \eta \cdot m_t / \sqrt{v_t}$ | 结合 momentum 和自适应学习率 |

## 5. Adam 的问题

### 5.1 泛化退化 (Generalization Gap)

Adam 在大模型 fine-tuning 场景常不如 SGD + Momentum。

原因分析：

1. **过强的自适应学习率**：$m_t/\sqrt{v_t}$ 在梯度稀疏时仍有过大波动
2. **不稳定的权重更新**：二阶矩估计引入额外噪声
3. **学习率尺度问题**：默认 $\eta=0.001$ 对大模型偏大

### 5.2 AdamW 的提出

**L2 正则 vs Weight Decay**：原始 Adam 对 L2 正则的梯度是 $2\lambda\theta$，但它被二阶矩归一化吸收了，导致实际正则效果弱。

AdamW (Decoupled Weight Decay)：

$$
\theta_t = \theta_{t-1} - \eta \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_{t-1} \right)
$$

权重衰减项独立于自适应学习率，正则效果更稳定。

### 5.3 数值稳定性问题

当 $v_t$ 很大时，$\sqrt{v_t}$ 可能 overflow。$\varepsilon$ 就是为防止此问题。

另外，若二阶矩估计持续很小（如梯度始终很小），学习率会过大，导致不稳定。

## 6. 变体与改进

### 6.1 AdamW (2020)

最广泛使用的 Adam 变体，已成 LLM 训练标准。

### 6.2 RAdam (Rectified Adam)

通过方差退火避免早期学习率过大：

$$
\rho_t = \rho_{\infty} - \frac{2t\beta_2^t}{1-\beta_2^t}
$$
$$
r_t = \sqrt{\frac{(\rho_t - 4)({\rho_t - 2}\rho_{\infty})}{{(\rho_{\infty} - 4)(\rho_{\infty} - 2)} \rho_t}}
$$

当 $\rho_t > 4$ 时使用修正学习率 $r_t \cdot \eta$，否则使用 $\eta$。

### 6.3 AdaBelief (2020)

将 $v_t$ 的估计改为：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) (g_t - m_t)^2
$$

即用" belief " in gradient 来调整二阶矩，在部分任务取得更好效果。

### 6.4 Lion (EvoLved Sign Momentum, 2023)

更简化的版本：

```
m_t = β1 · m_{t-1} + (1-β1) · g_t
u_t = sign(m_t)  # 用符号函数代替归一化
θ_t = θ_{t-1} - η · u_t
```

仅需存储 $m_t$，无需 $v_t$。内存减半，泛化能力有时优于 AdamW。

## 7. LLM 训练中的 Adam

### 7.1 混合精度训练 + Adam

通常使用 FP16/BF16 前向/反向，Adam 状态 (m, v) 保持 FP32 以保证精度：

```
for p in model.parameters():
    if p.requires_grad:
        # 主权重: BF16
        # Adam states: FP32
        # Gradient: BF16 (但通常用 bucket 累积到 FP32)
```

### 7.2 分页优化 (PagedAdam)

DeepSpeed ZeRO 优化：分片存储 $m$ 和 $v$ 到多卡，而非每卡复制完整状态。

### 7.3 学习率调度

Adam 对学习率调度敏感，常见策略：

- **Warmup**：5-6% 步数内从 0 上升到峰值（如 GPT-3 用 375M 步 warmup）
- **Cosine Decay**：峰值后余弦退火到约峰值的 10%
- **Linear Warmup + Cosine Decay**：先用线性 warmup，再用余弦

## 8. 代码实现

```python
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [torch.zeros_like(p) for p in params]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in params]  # 二阶矩
        self.t = 0  # 步数

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            # 一阶矩更新
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # 二阶矩更新
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g

            # 偏置校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 权重更新
            if self.weight_decay > 0:
                # AdamW: decoupled weight decay
                p.data -= self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps)
                                     + self.weight_decay * p.data)
            else:
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

## 9. 参考

- Adam 原始论文: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2014)
- AdamW: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- RAdam: [On the Variance of Adaptive Learning Rate](https://arxiv.org/abs/1908.03265)
- Lion: [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)