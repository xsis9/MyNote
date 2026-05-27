# DAM-VLA: Dynamic Action Model-Based Vision-Language-Action Framework

**来源**：arXiv（具体编号未提供）
**作者**：Xiongfeng Peng, Jiaqian Yu, Dingzhe Li, Yixiang Jin, Lu Xu, Yamin Mao, Chao Zhang, Weiming Li, Sujin Jang, Dongwook Lee, Daehyun Ji

---

## 1. 背景与问题

### 1.1 当前 VLA 方法的局限

当前 Vision-Language-Action (VLA) 框架大多从预训练的 VLMs 适配而来，存在以下问题：

1. **CoT 推理方法**（RT-H、RT-Affordance、ECoT）：引入大量额外推理 tokens（如 OpenVLA 7 个，ECoT 350 个），显著增加推理时间和控制频率

2. **扩散 VLA 方法**（π0、TinyVLA、RDT-1B、RoboDual、CogACT、HybridVLA）：在 VLM 后附加独立的扩散头，仅依赖 VLM 提取的特征，限制了对更丰富多模态信息的整合

3. **核心矛盾**：难以同时实现**任务通用性**和**精细操作的专用精度**

### 1.2 臂运动与夹爪操控的关键区别

论文识别出三个关键区别（以"将胡萝卜放到盘子上"任务为例）：

1. **路径约束（Path Constraints）**：
   - 臂运动：相对无约束，机器人可沿多条轨迹到达目标
   - 夹爪操控：高度约束，需要精确的抓取姿势才能成功

2. **视觉注意力（Visual Attention）**：
   - 臂运动：依赖全局场景理解
   - 夹爪操控：需要细粒度、局部的视觉注意力

3. **数据集表示（Dataset Representation）**：
   - 数据集中臂运动轨迹远多于夹爪操控
   - 尽管夹爪操控样本较少，但对任务成功至关重要，且通常更复杂

---

## 2. 方法

### 2.1 整体架构

给定语言指令 $l$ 和时刻 $t$ 的视觉观测 $\mathbf{o}_t$，模型 $\pi$ 预测时序动作序列：
$$[\mathbf{a}_t, \mathbf{a}_{t+1}, ..., \mathbf{a}_{t+N}] = \pi(l, \mathbf{o}_t)$$

动作空间：$\mathbf{a}_t = [\delta\mathbf{x}, \delta\boldsymbol{\theta}, s^{grip}]$，共 7 DoF：
- $\delta\mathbf{x}$：末端执行器相对平移
- $\delta\boldsymbol{\theta}$：旋转变化
- $s^{grip} \in \{0, 1\}$：夹爪开/闭状态

### 2.2 三大核心组件

**1. Vision-Language Model（视觉语言模型）**

视觉模型处理 RGB 图像输入，输出：
- **视觉 tokens** $\mathbf{f}^{vis}$：来自 DINOv2 和 SigLIP（预训练于互联网规模图像数据）
- **class token** $\mathbf{f}^{cls}$：用于需要全局注意力的臂运动模型
- **register token** $\mathbf{f}^{reg}$：用于需要局部注意力的夹爪操控模型

大语言模型（LLaMA-2）负责整合视觉信息和语言指令，输出：
- **cognition latent** $f^{cog}$：来自 LLM 最后一层，用于动作预测
- **reasoning latent** $\mathbf{f}^{rea}$：来自 LLM 第二层，用于动作路由

**2. Action Routing Module（动作路由模块）**

利用 VLM 的推理能力学习权重 $w$，监督信号为标记权重 $\hat{w} \in \{0, 1\}$：
- $\hat{w} = 0$：臂运动阶段
- $\hat{w} = 1$：夹爪操控阶段

标记规则：当机器人夹爪状态从开到闭或从闭到开时，判定为从臂运动转为夹爪操控。

通过交叉熵损失监督：
$$L_{class} = || -(\hat{w}\log(w) + (1-\hat{w})\log(1-w)) ||^1$$

**3. Dynamic Action Model（动态动作模型）**

双头扩散模型，结合：
- 高层 VLM cognition latent $f^{cog}$
- 低层视觉 tokens（class token 或 register token）

两个独立动作模型：
- **臂运动模型**：使用 class token $\mathbf{f}^{cls}$（需要全局注意力）
- **夹爪操控模型**：使用 register token $\mathbf{f}^{reg}$（需要局部注意力）

损失函数（使用 Mahalanobis 距离加权）：
$$L_{move} = ||\mathbf{n}_i^{move} - \hat{\mathbf{n}}^{move}||_{\Sigma\hat{\mathbf{w}}^{move}}^2$$
$$L_{mani} = ||\mathbf{n}_i^{mani} - \hat{\mathbf{n}}^{mani}||_{\Sigma\hat{\mathbf{w}}^{mani}}^2$$

总损失：
$$L = w_1 \cdot L_{move} + w_2 \cdot L_{mani} + w_3 \cdot L_{class}$$
其中 $w_1 = 1.0, w_2 = 1.0, w_3 = 0.0001$

### 2.3 双尺度动作加权机制

**目的**：增强区分臂运动和夹爪操控的鲁棒性，通过从全局（轨迹级）和局部（动作 chunk 级）两个角度自适应调节不同动作类型的重要性。

**轨迹级权重（Trajectory-level Weights）**：

基于机器人夹爪的二元状态变化，将整个任务轨迹分割为臂运动和夹爪操控阶段。对每个夹爪操控过程 $k$，采用非对称高斯分布：
$$\{\mathcal{N}(u, \sigma_l^2), \mathcal{N}(\bar{u}, \sigma_r^2)\}$$

两个分布共享相同均值 $u$（夹爪状态转换的时间中点）。前沿方差更大（$\sigma_l = 6$），后沿方差更小（$\sigma_r = 2$），反映动作模型在状态变化前需要更高精度和监督焦点的先验。

聚合轨迹权重：
$$\Delta\mathbf{w}^t = \text{Norm}(\sum_k \mathbf{w}_k^t)$$

**动作 chunk 级权重（Action-chunk-level Weights）**：

考虑动作序列的内在时间不确定性。预测置信度通常随时距增加而衰减，采用指数衰减：
$$w_j^a = \gamma^j$$
其中 $j$ 是动作 chunk 内索引，$\gamma = 0.8$ 是衰减因子。

**多尺度整合**：
$$\mathbf{w}^{move} = (1 - \mathbf{w}^t) \odot \mathbf{w}^a$$
$$\mathbf{w}^{mani} = \mathbf{w}^t \odot \mathbf{w}^a$$

### 2.4 推理阶段

动态动作模型根据预测权重 $w$ 选择执行对应动作模型：
- 若 $w < 0.5$：执行臂运动模型，使用 cognition latent $\mathbf{f}^c$ 和全局 class token $\mathbf{f}^{cls}$ 预测多步动作序列
- 若 $w \geq 0.5$：执行夹爪操控模型，使用 cognition latent $f^c$ 和局部 register token $\mathbf{f}^{reg}$ 预测动作序列

两个模型都接收随机噪声 $\boldsymbol{\omega}_n^{rand}$ 作为输入以支持扩散过程。

---

## 3. 训练配置

### 3.1 预训练

使用 Open X-Embodiment Dataset 中的两个主要子集：
- **Fractal**
- **Bridge-DataV2**

训练配置：
- 学习率：$2 \times 10^{-5}$（常数）
- Batch size：256
- 硬件：8 NVIDIA H100 GPUs
- 训练时长：约 2 天

### 3.2 微调

**模拟环境**：FurnitureBench 基准，接触丰富和长时域操作任务
- 使用 "One-Leg" 装配任务的 500 条专家轨迹微调

**真实环境**：Franka 机器人执行 pick-and-place
- 收集 50 条演示轨迹
- 学习率和 batch size 与预训练相同

---

## 4. 实验结果

### 4.1 SIMPLER 模拟评估

**Google 机器人（VA 设置）**：

| 方法 | PCC | MN | OCD | ODPA | 平均 |
|------|-----|-----|-----|------|------|
| RT-1 | 90% | 46% | 35% | 3% | 44% |
| RT-2-X | 82% | 79% | 35% | 21% | 54% |
| OpenVLA | 64% | 64% | 19% | 1% | 37% |
| CogACT | 96% | 84% | 29% | 40% | 62% |
| **DAM-VLA** | **98%** | **74%** | **68%** | **84%** | **81%** |

**Google 机器人（VM 设置）**：

| 方法 | PCC | MN | OCD | ODPA | 平均 |
|------|-----|-----|-----|------|------|
| OpenVLA | 14% | 51% | 48% | 0% | 28% |
| CogACT | 92% | 82% | 75% | 39% | 72% |
| π₀ | 89% | 81% | 55% | 53% | 70% |
| **DAM-VLA** | **96%** | **84%** | **75%** | **78%** | **83%** |

**WidowX 机器人（VM 设置）**：

| 方法 | SoT | CoP | GoY | EiB | 平均 |
|------|-----|-----|-----|------|------|
| OpenVLA | 4% | 0% | 0% | 4% | 2% |
| CogACT | 63% | 50% | 25% | 71% | 52% |
| π₀ | 62% | 59% | 24% | 81% | 57% |
| **DAM-VLA** | **88%** | **71%** | **25%** | **100%** | **71%** |

### 4.2 FurnitureBench 长时域任务

**"One-Leg" 装配任务各步骤成功率**：

| 方法 | 步骤1 | 步骤2 | 步骤3 | 步骤4 | 步骤5（最终） |
|------|-------|-------|-------|-------|--------------|
| OpenVLA | 96% | 94% | 78% | 53% | 29% |
| CogACT | 98% | 96% | 96% | 56% | 42% |
| **DAM-VLA** | **100%** | **100%** | **100%** | 62% | **56%** |

### 4.3 真实世界评估

| 方法 | ID（分布内） | OOD（分布外） | 平均 |
|------|-------------|--------------|------|
| CogACT | 65.7% | 60.0% | 62.9% |
| **DAM-VLA** | **91.4%** | **82.2%** | **86.8%** |

---

## 5. 消融实验

在 WidowX（VM）和 Google（VM/VA）机器人上的消融结果：

| VT | ACW | TW | DAM | DL | Google(VM) | Google(VA) | WidowX(VM) | 平均 |
|----|----|----|----|----|------------|------------|------------|------|
| - | - | - | - | - | 64% | 61% | 50% | 58% |
| √ | √ | - | - | - | 78% | 68% | 53% | 66% |
| √ | √ | √ | - | - | 76% | 63% | 51% | 63% |
| √ | √ | √ | √ | - | 82% | 72% | 43% | 66% |
| √ | √ | √ | √ | √ | 83% | 81% | 71% | **78%** |
| - | √ | √ | √ | √ | 84% | 75% | 58% | 73% |
| - | - | - | √ | √ | 66% | 64% | 49% | 60% |

**缩写说明**：
- VT：Visual Tokens（视觉 class 和 register tokens）
- ACW：Action Chunk Weights（动作 chunk 级权重）
- TW：Trajectory Weights（轨迹级权重）
- DAM：Dynamic Action Model（臂运动模型 + 夹爪操控模型）
- DL：Dual Latents（从 LLM 不同层提取的 reasoning 和 cognition latents）

---

## 6. 核心贡献

1. **动作路由**：VLM 引导的路由器解读任务相关的视觉和语言线索，选择合适的动作模型

2. **动态动作模型**：双头扩散模型将 VLM 高层认知与低层视觉信息结合，预测动作序列

3. **双尺度动作加权**：轨迹级和动作 chunk 级双重加权机制，实现臂运动和夹爪操控模型间的动态协调

4. **广泛评估**：在 SIMPLER、FurnitureBench 模拟环境及真实世界 pick-and-place 任务中均超越现有 VLA 方法