# X-WAM: Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising

**来源**：arXiv:2602.15922
**作者**：Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, Huaping Liu（清华大学 & 小米机器人）
**项目主页**：https://sharinka0715.github.io/X-WAM/

---

## 1. 背景与问题

### 1.1 当前范式

当前机器人基础模型分为两类：

1. **策略模型**（Policy Models）：Vision-Language-Action (VLA) 模型，直接预测机器人动作，擅长指令跟随和语义推理，但缺乏几何直觉和对物理世界的感知

2. **世界模型**（World Models）：专注于生成未来观测（视频生成），不直接产生可执行的动作

两类模型各自独立发展，限制了跨任务协同和表征效率。

### 1.2 已有统一尝试的问题

近期统一世界动作模型（World Action Models, WAMs）尝试联合建模视频生成和动作预测，但仍存在两个关键局限：

- **局限一**：现有统一模型仅在 2D 像素空间建模，缺乏显式的 3D 空间感知和几何 grounding
- **局限二**：虽然一些 WAMs 通过解耦视频和动作的采样 timestep 来平衡视频生成质量和动作解码效率，但训练时使用独立噪声采样，导致训练分布与推理分布不匹配

### 1.3 X-WAM 的目标

X-WAM 是一个统一 4D 世界动作模型，在单一架构内同时实现四个目标：
1. 高保真视频生成
2. 3D 空间重建
3. 高策略成功率
4. 高效动作执行

---

## 2. 模型架构

### 2.1 整体输入输出

X-WAM 以以下内容为输入：
- 语言指令 $c$
- 初始本体感受状态 $s_0$
- 多视角初始 RGB 观测 $O_0$

输出：
- 未来 RGB 视频 $O_{1:H}$
- 深度视频 $D_{1:H}$
- 本体感受状态 $s_{1:H}$
- 动作 $a_{1:K}$

其中 $H$ 和 $K$ 分别为视频/状态和动作的预测视野。

### 2.2 架构细节

**基础模型**：基于预训练视频生成 Diffusion Transformer，具体使用 Wan2.2-TI2V-5B。

**模态编码**：
- RGB 视频：通过原始因果 VAE 编码器 $\mathcal{E}$ 编码为潜在表示 $\mathbf{z}_O = \mathcal{E}(O)$
- 本体感受状态和动作：通过可学习的 MLP 投影到潜在空间：$\mathbf{z}_s = \text{MLP}_s(s)$，$\mathbf{z}_a = \text{MLP}_a(a)$
- 解码时通过对称 MLP：$\hat{s} = \text{MLP}_s^{dec}(\mathbf{z}_s)$，$\hat{a} = \text{MLP}_a^{dec}(\mathbf{z}_a)$

**统一去噪序列**：三种模态拼接为统一序列：
$$\mathbf{Z} = \left[\mathbf{z}_{O_0}, \mathbf{z}_{O_{1:H}}, \mathbf{z}_{s_0}, \mathbf{z}_{s_{1:H}}, \mathbf{z}_{a_{1:K}}\right]$$

初始观测和状态在去噪过程中保持固定，噪声 timestep 设为 $t = 0$（视为干净样本）。

### 2.3 时间设计

给定 1 个条件 RGB 帧和 1 个初始状态：
- 预测 $H = 8$ 个未来 RGB 帧
- 预测 $H = 8$ 个未来状态
- 预测 $K = 32$ 个未来动作

**设计原因**：不同模态的时间需求不同——动作需要更高频率以保证平滑响应，RGB 帧和状态可以用较低频率预测。状态与视频帧时间对齐以支持逐帧多视角 RGB-D 融合进行 3D 重建，而动作在相同时间跨度内以 $K/H = 4\times$ 视频帧率均匀分布。

### 2.4 多视角和位置编码

原视频扩散模型使用 3D RoPE 编码时间和空间位置。为支持多视角而不破坏预训练位置编码，给每个视角的 token 添加可学习的视角嵌入。本体感受状态和动作在时间维度上应用与视频 token 相同的时域 RoPE。

### 2.5 相机姿态处理

X-WAM 需要相机姿态来从多视角 RGB-D 输出重建 3D 表征（如点云）。观察到机械臂设置中的相机分为两类：
- **静态相机**（第一人称和第三人称视角）：姿态在任务执行过程中保持不变
- **动态相机**（腕部相机）：刚性地连接在机械臂上，通过固定手眼标定矩阵依赖于机械臂模型

因此 X-WAM 预测末端执行器位姿 $\mathbf{T}_{ee} \in SE(3)$，并通过固定手眼标定矩阵 $\mathbf{T}_{h2e}$ 导出腕部相机位姿：
$$\mathbf{T}_{wrist} = \mathbf{T}_{ee} \cdot \mathbf{T}_{h2e}$$

---

## 3. 轻量化深度适应模块

### 3.1 设计动机

将深度信息注入模型面临两个挑战：
- 简单沿序列维度拼接深度图会使序列长度翻倍，导致注意力成本二次增长
- 沿通道维度融合会将输入分布偏离预训练流形，大幅增加学习难度

### 3.2 具体方法

不扩张去噪序列，X-WAM 通过简单复制预训练 DiT 的最后 M 个块来构建专用深度预测分支。

设模型有 N 个 DiT 块，复制最后 M 个块作为深度分支。经过共享的前 $N-M$ 个块产生隐状态 $\mathbf{H}$ 后，深度分支和主分支初始化为：
$$\mathbf{Z}_D^{(0)} = \mathbf{Z}_m^{(0)} = \bar{\mathbf{H}}$$

然后以交错方式执行。在每层 $j \in \{1, \dots, M\}$：
$$\mathbf{Z}_D^{(j)} = \text{DephBlock}_j\left(\mathbf{Z}_D^{(j-1)} \mid \mathbf{Z}_m^{(j-1)}\right), \quad \mathbf{Z}_m^{(j)} = \text{DitBlock}_{N-M+j}\left(\mathbf{Z}_m^{(j-1)}\right)$$

深度块通过 cross-attention 从主分支的输入读取信息，但主分支不受深度 token 影响。这种非对称连接称为 **unilateral attention**：深度分支可以读取主分支，但反之不行，从而严格保护预训练权重完整性。

### 3.3 深度监督

深度分支使用 MSE 损失回归当前视频帧的逆深度。推理时此辅助深度分支不需要参与每个去噪步骤，可以灵活开关，大幅减少 rollout 开销。

---

## 4. 异步噪声采样（ANS）

### 4.1 问题

现有方法解耦视频和动作的噪声 timestep 以实现更快的动作解码。但独立采样每个模态的 timestep 会引入训练配置中从未出现在测试时的情况（如 $t_O < t_a$），降低训练效率。

### 4.2 异步推理

推理时 ANS 为视频和动作应用异步去噪 timestep。分配 $T_a$ 步用于本体感受状态和动作去噪，$T_O$ 步用于视频（$T_a < T_O$）。两个模态都从纯噪声开始，分别以步长 $\frac{1}{T_a}$ 和 $\frac{1}{T_O}$ 去噪。经过 $T_a$ 步后，获得无噪声动作可立即发送到下游机器人执行。如需完整清晰视频，继续剩余 $T_O - T_a$ 步，期间动作作为干净模态不再去噪。

### 4.3 训练时的耦合噪声采样

为实现异步推理行为，训练时也对视频和动作应用不同噪声 timestep。但与之前独立采样的方法不同，X-WAM 将视频和动作的噪声水平放入联合分布进行耦合采样。

形式上，联合噪声水平 $(t_O, t_a)$ 从以下混合分布中采样：
$$\left(t_O, t_a\right) \sim \begin{cases} t_a = 0, t_O \sim \mathrm{U}(0, 1) & \text{ w.p. } p, \\ t_a \sim \mathrm{U}(0, 1), t_O = t_a + (1 - t_a) \cdot b, b \sim \text{Beta}(1.5, 1) & \text{ w.p. } 1 - p \end{cases}$$

第一项对应动作条件视频生成（无噪声动作），第二项表示异步联合生成。Beta(1.5, 1) 分布在 $[t_a, 1]$ 上重缩放，使 $t_O$ 偏向更高噪声水平，反映视频通常需要比动作更多的去噪步骤。关键点：$t_O$ 在 $t_a$ 上有条件采样，使它们是相依的而非独立随机变量。

---

## 5. 训练

### 5.1 训练框架

与预训练 Wan2.2-5B 模型一致，X-WAM 使用 flow matching 框架训练。模型 $f_\theta$ 被训练在给定 timestep t 的噪声输入下预测速度场 $\mathbf{v} = \epsilon - \mathbf{z}^0$。

对于模态 $m \in \{O, s, a\}$，速度预测损失为：
$$\mathcal{L}_m = \left\| f_\theta^m(\mathbf{z}_m^{t_m}, t_m) - (\boldsymbol{\epsilon}_m - \mathbf{z}_m^0) \right\|^2$$

深度分支监督使用直接 MSE 回归损失：
$$\mathcal{L}_{depth} = \left\| \hat{D} - D^* \right\|^2$$

总损失：
$$\mathcal{L}_{total} = \mathcal{L}_O + \lambda_s \mathcal{L}_s + \lambda_a \mathcal{L}_a + \lambda_D \mathcal{L}_{depth}$$

### 5.2 状态和动作表示

为统一异构单臂和双臂机器人，定义基于末端执行器位姿的通用状态和动作接口：
- **状态**：16 维绝对向量 =（位置3 + 四元数4 + 夹爪1）× 2 臂
- **动作**：14 维相对向量 =（$\Delta position_3$ + $\Delta axis-angle_3$ + $gripper_1$）× 2 臂

单臂机器人只监督状态的前 8 维和动作的前 7 维。

### 5.3 预训练数据

在超过 5,800 小时的机器人数据上训练，包括真实机器人和模拟环境数据集。所有数据经过预处理和过滤，统一到一致的坐标系和表示。

| 数据集 | 来源 | Episodes | 时长(h) |
|--------|------|----------|---------|
| AgibotWorld-Beta | Real | 866,562 | 2,221.5 |
| DROID | Real | 74,734 | 280.3 |
| InternA1-Aloha | Sim | 184,803 | 1,337.3 |
| InternA1-Genie1 | Sim | 50,638 | 174.0 |
| InternA1-Lift2 | Sim | 231,018 | 1,464.7 |
| RoboCasa MimicGen | Sim | 56,771 | 282.4 |
| RoboTwin 2.0 | Sim | 27,500 | 113.7 |
| **总计** | - | 1,492,026 | 5,873.9 |

### 5.4 训练配置

- 预训练：256 NVIDIA H20 GPUs，per-GPU batch size 8（总 batch size 2,048）
- Optimizer: AdamW，峰值学习率 $1 \times 10^{-4}$，1000 步线性 warmup + 余弦衰减到 0
- 训练步数：40,000 步
- 预测视野：$H = 8$
- 损失权重：$\lambda_s = 1.0, \lambda_a = 1.0, \lambda_D = 1.0$
- 复制深度块数：$M = 10$
- ANS 动作条件概率：$p = 0.5$

- Benchmark 微调：32 NVIDIA H20 GPUs，per-GPU batch size 4（总 batch size 128），学习率 $3 \times 10^{-5}$，20,000 步

---

## 6. 推理

### 6.1 异步去噪配置

使用异步去噪，$T_a = 10$ 步动作去噪，$T_O = 50$ 步视频去噪，采用 UniPC 调度器。

### 6.2 推理伪代码（算法2）

```
输入：条件 (O₀, s₀, c)，视频步数 T_O，动作步数 T_a（T_a < T_O）

1: z_O, z_s, z_a ~ N(0, I)
2: 初始化调度器：S_O with T_O 步，S_a with T_a 步

3: for k = 1 to T_O do
4:    从 S_O 获取当前 timestep t_O
5:    if k ≤ T_a then
6:        从 S_a 获取当前 timestep t_a
7:        预测 v̂_O, v̂_s, v̂_a ← DENOISE(z_O, z_s, z_a, t_O, t_a, O₀, s₀, c)
8:        z_O ← S_O.step(z_O, v̂_O); z_s ← S_a.step(z_s, v̂_s); z_a ← S_a.step(z_a, v̂_a)
9:    else
10:       预测 v̂_O ← DENOISE(z_O, z_s, z_a, t_O, 0, O₀, s₀, c)
11:       z_O ← S_O.step(z_O, v̂_O)
12:   end if
13: end for

14: return z_O, z_s, z_a  # 动作在 T_a 步后可用，视频在 T_O 步后完成
```

### 6.3 单步去噪伪代码（算法1）

```
输入：噪声视频 latent z_O^{t_O}，噪声状态 z_s^{t_a}，噪声动作 z_a^{t_a}，视频 timestep t_O，动作 timestep t_a，初始观测 O₀，初始状态 s₀，语言指令 c

1: z_{O₀} ← CausalVAE(O₀); z_{s₀} ← MLP_s(s₀)  # 用 t=0 编码条件
2: Z ← Concat(z_{O₀}, z_O^{t_O}, z_{s₀}, z_s^{t_a}, z_a^{t_a})
3: 添加可学习的视角嵌入到 Z
4: for i = 1 to N-M do
5:     Z ← DiTBlock_i(Z)  # 共享躯干
6: end for
7: Z_m ← Z; Z_D ← Z  # 初始化主分支和深度分支
8: for j = 1 to M do
9:     Z_D ← DepthBlock_j(Z_D | Z_m)  # 深度注意主分支输入
10:    Z_m ← DiTBlock_{N-M+j}(Z_m)  # 主分支
11: end for
12: v̂_O, v̂_s, v̂_a ← Head_main(Z_m)
13: D̂ ← Head_depth(Z_D)  # 回归逆深度
14: return v̂_O, v̂_s, v̂_a, D̂
```

### 6.4 Real-Time Chunking (RTC)

采用 Real-Time Chunking 方法将去噪计算与动作执行重叠，使机器人能够以 15 Hz 控制频率无缝实时部署。

---

## 7. 实验结果

### 7.1 策略评估

**RoboCasa（24 个操作任务）**：

| 方法 | 平均成功率 |
|------|-----------|
| π₀ (VLA) | 62.5% |
| GR00T-N1.5 (VLA) | 64.1% |
| UWM (WAM) | 60.8% |
| DreamZero (WAM) | 62.4% |
| Cosmos Policy (WAM) | 67.1% |
| **X-WAM** | **79.2%** |

**RoboTwin 2.0（50 个双臂操作任务）**：

| 方法 | Clean | Randomized |
|------|-------|-----------|
| π₀ (VLA) | 65.9% | 58.4% |
| π₀.₅ (VLA) | 82.7% | 76.8% |
| UWM (WAM) | 81.7% | 78.6% |
| GigaWorld-Policy (WAM) | 87.0% | 85.0% |
| Motus (WAM) | 88.7% | 87.0% |
| **X-WAM** | **89.8%** | **90.7%** |

### 7.2 4D 重建质量（RoboCasa）

| 方法 | RGB PSNR↑ | RGB SSIM↑ | RGB LPIPS↓ | Depth AbsRel↓ | Depth δ₁↑ | Point Cloud CD↓ |
|------|-----------|-----------|------------|---------------|-----------|------------------|
| DreamZero + DA3 | 21.12 | 0.7788 | 0.1580 | 0.1362 | 0.8594 | 0.0680 |
| Robot4DGen | 22.67 | 0.8207 | 0.1026 | 0.0736 | 0.9443 | 0.0134 |
| X-WAM w/o depth + DA3 | 23.09 | 0.8916 | 0.0548 | 0.1045 | 0.9089 | 0.0401 |
| **X-WAM** | **23.46** | **0.8942** | **0.0513** | **0.0349** | **0.9738** | **0.0049** |

### 7.3 消融实验

**深度架构设计**：

| 变体 | 成功率 | 延迟(ms) | PSNR↑ | CD↓ |
|------|--------|----------|-------|-----|
| No depth | 63.0% | 1033 | 23.09 | - |
| Sequence concatenation | 68.7% | 1888 | 23.60 | 0.0037 |
| Channel concatenation | 64.2% | 1266 | 23.20 | 0.0052 |
| **Interleaved branch (Ours)** | **67.8%** | **1033** | **23.46** | **0.0049** |

**噪声调度策略**：

| 配置 | 成功率 | 延迟(ms) | PSNR↑ | CD↓ |
|------|--------|----------|-------|-----|
| Sync train + Sync infer | 66.4% | 4665 | 23.48 | 0.0041 |
| Decoupled train + Sync infer | 66.3% | 4665 | 23.17 | 0.0050 |
| Decoupled train + Async infer | 67.2% | 1033 | 22.60 | 0.0061 |
| **ANS train + Async infer** | **67.8%** | **1033** | **23.46** | **0.0049** |

---

## 8. 局限性

1. **上下文窗口固定**：当前框架只处理固定长度的观测上下文，不包含历史信息或自回归 rollout，可能影响长时域操作场景中的任务进度理解

2. **推理延迟较高**：作为联合生成高维视频和低维动作的统一模型，相比专用策略模型有更高推理延迟。使用 8 步去噪时每动作块约 300ms，虽然 Real-Time Chunking 实现了无缝部署，但额外推理延迟仍可能降低策略性能

---

## 9. 核心贡献总结

1. **轻量化深度适应模块**：通过复制预训练 DiT 最后 M 个块作为交错深度分支，实现高质量空间重建而不增加序列长度或破坏预训练视觉先验

2. **异步噪声采样（ANS）**：通过联合分布采样和对齐训练推理噪声分布，提高训练效率的同时最大化动作解码速度和视频生成质量

3. **统一 4D 框架**：单一框架可联合优化策略执行、视频生成和空间重建，在 RoboCasa 和 RoboTwin 2.0 基准测试中一致超越所有基线