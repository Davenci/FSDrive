# MoVQGAN 模型结构分析

本文档从深度学习、线性代数与数学角度，对 MoVQGAN 的前向传播逻辑进行梳理。按模块归纳，细化到线性投影层、激活函数与归一化函数，并附带公式说明。文末整理 MoVQGAN 与 VQGAN 的核心区别。

---

## 目录

1. [基础算子定义](#1-基础算子定义)
2. [Encoder（编码器）](#2-encoder编码器)
3. [VectorQuantizer2（向量量化器）](#3-vectorquantizer2向量量化器)
4. [MOVQDecoder（解码器）](#4-movqdecoder解码器)
   - 4.1 [SpatialNorm（空间自适应归一化）](#41-spatialnorm空间自适应归一化)
   - 4.2 [ResnetBlock（含 zq 条件）](#42-resnetblock含-zq-条件)
   - 4.3 [AttnBlock（含 zq 条件）](#43-attnblock含-zq-条件)
   - 4.4 [MOVQDecoder 整体前向](#44-movqdecoder-整体前向)
5. [MOVQ 顶层前向传播](#5-movq-顶层前向传播)
6. [MoVQGAN 与 VQGAN 的核心区别](#6-movqgan-与-vqgan-的核心区别)

---

## 1. 基础算子定义

### 1.1 Swish 激活函数

$$\text{swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

- $x$：输入张量，形状任意
- $\sigma(\cdot)$：sigmoid 函数
- 操作类型：**逐元素乘法**（element-wise multiply）

```python
def nonlinearity(x):
    return x * torch.sigmoid(x)
```

### 1.2 GroupNorm 归一化

$$\text{GroupNorm}(x) = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} \cdot \gamma + \beta$$

- $x \in \mathbb{R}^{B \times C \times H \times W}$：输入特征图
- $g$：组索引，共 32 组，每组包含 $C/32$ 个通道
- $\mu_g, \sigma_g^2$：第 $g$ 组在 $(C/32) \times H \times W$ 维度上的均值与方差
- $\gamma, \beta \in \mathbb{R}^C$：可学习的仿射参数（`affine=True`）
- $\epsilon = 10^{-6}$：数值稳定项
- 操作类型：通道内归一化 + 仿射变换

---

## 2. Encoder（编码器）

> 文件：`movqgan/modules/vqvae/vqvae_blocks.py`，类 `Encoder`

MoVQGAN 的编码器与 VQGAN 完全相同，未做修改。

### 2.1 整体结构

```
输入 x: [B, 3, H, W]
  │
  ▼
conv_in: Conv2d(3, ch, 3×3, pad=1)          → [B, ch, H, W]
  │
  ▼
下采样阶段（num_resolutions 层）
  每层：
    num_res_blocks × ResnetBlock
    （若当前分辨率在 attn_resolutions 中）× AttnBlock
    Downsample（stride=2）→ 分辨率减半
  │
  ▼
中间层（bottleneck）
  ResnetBlock → AttnBlock → ResnetBlock
  │
  ▼
norm_out: GroupNorm(32)
swish
conv_out: Conv2d(block_in, z_channels, 3×3, pad=1)
  │
  ▼
输出 h: [B, z_channels, H/f, W/f]   （f = 2^(num_resolutions-1)）
```

### 2.2 ResnetBlock（Encoder 内，标准版）

输入：$x \in \mathbb{R}^{B \times C_{in} \times H \times W}$，无 `zq` 条件

$$\begin{aligned}
h &= \text{GroupNorm}(x) \\
h &= \text{swish}(h) \\
h &= \text{Conv}_{3\times3}^{C_{in} \to C_{out}}(h) \\
h &= \text{GroupNorm}(h) \\
h &= \text{swish}(h) \\
h &= \text{Dropout}(h) \\
h &= \text{Conv}_{3\times3}^{C_{out} \to C_{out}}(h) \\
x' &= \begin{cases} \text{Conv}_{1\times1}^{C_{in} \to C_{out}}(x) & \text{if } C_{in} \neq C_{out} \\ x & \text{otherwise} \end{cases} \\
\text{output} &= x' + h \quad \text{（残差加和）}
\end{aligned}$$

### 2.3 AttnBlock（Encoder 内，标准版）

输入：$x \in \mathbb{R}^{B \times C \times H \times W}$

$$\begin{aligned}
h &= \text{GroupNorm}(x) \\
Q &= \text{Conv}_{1\times1}(h),\quad K = \text{Conv}_{1\times1}(h),\quad V = \text{Conv}_{1\times1}(h) \\
&\quad Q,K,V \in \mathbb{R}^{B \times C \times H \times W}
\end{aligned}$$

展平空间维度：

$$Q' = Q.\text{reshape}(B, C, HW).\text{permute}(0,2,1) \in \mathbb{R}^{B \times HW \times C}$$
$$K' = K.\text{reshape}(B, C, HW) \in \mathbb{R}^{B \times C \times HW}$$

注意力权重（矩阵乘法）：

$$W = Q' \cdot K' \in \mathbb{R}^{B \times HW \times HW}, \quad W \leftarrow W \cdot C^{-0.5}$$
$$A = \text{softmax}(W, \text{dim}=-1)$$

加权求和（矩阵乘法）：

$$V' = V.\text{reshape}(B, C, HW)$$
$$h' = V' \cdot A^T \in \mathbb{R}^{B \times C \times HW} \to \text{reshape} \to \mathbb{R}^{B \times C \times H \times W}$$

输出投影与残差：

$$\text{output} = x + \text{Conv}_{1\times1}(h') \quad \text{（残差加和）}$$

### 2.4 Downsample

```
with_conv=True:
  pad (0,1,0,1) → Conv2d(C, C, 3×3, stride=2, pad=0)
with_conv=False:
  AvgPool2d(kernel=2, stride=2)
```

---

## 3. VectorQuantizer2（向量量化器）

> 文件：`movqgan/modules/vqvae/quantize.py`，类 `VectorQuantizer2`

### 3.1 码本

$$E \in \mathbb{R}^{N \times D}$$

- $N$：码本大小（`n_e`）
- $D$：嵌入维度（`e_dim`）
- 初始化：$\text{Uniform}(-1/N,\ 1/N)$

### 3.2 前向传播

**输入**：$z \in \mathbb{R}^{B \times D \times H \times W}$（来自 `quant_conv` 输出）

**步骤 1：重排与展平**

$$z \leftarrow \text{rearrange}(z,\ b\ c\ h\ w \to b\ h\ w\ c) \in \mathbb{R}^{B \times H \times W \times D}$$
$$z_f = z.\text{view}(-1, D) \in \mathbb{R}^{BHW \times D}$$

**步骤 2：L2 距离计算（展开平方技巧）**

$$d_{ij} = \|z_f^{(i)} - e_j\|^2 = \|z_f^{(i)}\|^2 + \|e_j\|^2 - 2\, z_f^{(i)} \cdot e_j$$

矩阵形式（矩阵乘法）：

$$D = \underbrace{\|z_f\|^2_{\text{row}}}_{BHW \times 1} + \underbrace{\|E\|^2_{\text{row}}}_{1 \times N} - 2\, \underbrace{z_f \cdot E^T}_{BHW \times N}$$

$$D \in \mathbb{R}^{BHW \times N}$$

**步骤 3：最近邻查找**

$$k^* = \arg\min_j D_{ij} \quad \forall i \in \{1,\ldots,BHW\}$$

$$z_q = E[k^*] \in \mathbb{R}^{BHW \times D} \to \text{view}(B, H, W, D)$$

**步骤 4：损失计算**（legacy 模式）

$$\mathcal{L}_{VQ} = \underbrace{\mathbb{E}\left[\|z_q.\text{detach}() - z\|^2\right]}_{\text{commitment loss}} + \beta \cdot \underbrace{\mathbb{E}\left[\|z_q - z.\text{detach}()\|^2\right]}_{\text{codebook loss}}$$

- commitment loss：推动编码器输出靠近码本
- codebook loss：推动码本向编码器输出靠近
- $\beta = 0.25$

**步骤 5：直通梯度估计（Straight-Through）**

$$z_q \leftarrow z + (z_q - z).\text{detach()}$$

前向值为 $z_q$，反向梯度直接传回 $z$（绕过 argmin 的不可微性）。

**步骤 6：重排输出**

$$z_q \leftarrow \text{rearrange}(z_q,\ b\ h\ w\ c \to b\ c\ h\ w) \in \mathbb{R}^{B \times D \times H \times W}$$

**输出**：$(z_q,\ \mathcal{L}_{VQ},\ k^*)$

---

## 4. MOVQDecoder（解码器）

> 文件：`movqgan/modules/vqvae/movq_modules.py`，类 `MOVQDecoder`

MoVQGAN 解码器接受两个输入：
- $z \in \mathbb{R}^{B \times z\_channels \times h \times w}$：经 `post_quant_conv` 映射后的特征
- $zq \in \mathbb{R}^{B \times embed\_dim \times h \times w}$：原始量化特征，作为空间条件信号贯穿整个解码过程

### 4.1 SpatialNorm（空间自适应归一化）

MoVQGAN 的核心创新模块。

**参数**：
- `norm_layer`：GroupNorm(32 groups)，对特征 $f$ 做标准归一化
- `conv_y`：$\text{Conv}_{1\times1}^{D_{zq} \to C_f}$，生成空间自适应 scale
- `conv_b`：$\text{Conv}_{1\times1}^{D_{zq} \to C_f}$，生成空间自适应 bias

**前向**：

输入 $f \in \mathbb{R}^{B \times C_f \times H_f \times W_f}$，$zq \in \mathbb{R}^{B \times D_{zq} \times H_{zq} \times W_{zq}}$

**步骤 1：空间对齐（最近邻插值）**

$$zq' = \text{interpolate}(zq,\ \text{size}=(H_f, W_f),\ \text{mode=nearest}) \in \mathbb{R}^{B \times D_{zq} \times H_f \times W_f}$$

**步骤 2：标准归一化**

$$\hat{f} = \text{GroupNorm}(f) \in \mathbb{R}^{B \times C_f \times H_f \times W_f}$$

**步骤 3：生成空间自适应参数（线性投影）**

$$\gamma = \text{Conv}_{1\times1}(zq') \in \mathbb{R}^{B \times C_f \times H_f \times W_f}$$
$$\beta_{sp} = \text{Conv}_{1\times1}(zq') \in \mathbb{R}^{B \times C_f \times H_f \times W_f}$$

**步骤 4：仿射调制（逐元素乘法 + 加法）**

$$\text{output} = \hat{f} \odot \gamma + \beta_{sp}$$

其中 $\odot$ 表示逐元素乘法（element-wise multiply）。

与标准 GroupNorm 的对比：

| | 标准 GroupNorm | SpatialNorm |
|---|---|---|
| $\gamma, \beta$ 来源 | 可学习标量参数 $\in \mathbb{R}^C$ | 由 $zq$ 动态生成，$\in \mathbb{R}^{B \times C \times H \times W}$ |
| 空间变化性 | 无（全局共享） | 有（每个位置不同） |
| 条件信息 | 无 | 量化码本特征 $zq$ |

### 4.2 ResnetBlock（含 zq 条件）

输入：$x \in \mathbb{R}^{B \times C_{in} \times H \times W}$，$zq \in \mathbb{R}^{B \times D_{zq} \times h \times w}$

$$\begin{aligned}
h &= \text{SpatialNorm}(x,\ zq) \\
h &= \text{swish}(h) \\
h &= \text{Conv}_{3\times3}^{C_{in} \to C_{out}}(h) \\
h &= \text{SpatialNorm}(h,\ zq) \\
h &= \text{swish}(h) \\
h &= \text{Dropout}(h) \\
h &= \text{Conv}_{3\times3}^{C_{out} \to C_{out}}(h) \\
x' &= \begin{cases} \text{Conv}_{1\times1}^{C_{in} \to C_{out}}(x) & \text{if } C_{in} \neq C_{out} \\ x & \text{otherwise} \end{cases} \\
\text{output} &= x' + h \quad \text{（残差加和）}
\end{aligned}$$

与 Encoder 中 ResnetBlock 的区别：归一化层由 `GroupNorm` 替换为 `SpatialNorm`，后者接受额外的 $zq$ 输入。

### 4.3 AttnBlock（含 zq 条件）

输入：$x \in \mathbb{R}^{B \times C \times H \times W}$，$zq \in \mathbb{R}^{B \times D_{zq} \times h \times w}$

$$\begin{aligned}
h &= \text{SpatialNorm}(x,\ zq) \\
Q &= \text{Conv}_{1\times1}(h),\quad K = \text{Conv}_{1\times1}(h),\quad V = \text{Conv}_{1\times1}(h)
\end{aligned}$$

后续注意力计算与 §2.3 相同，最终：

$$\text{output} = x + \text{Conv}_{1\times1}(h') \quad \text{（残差加和）}$$

区别：归一化层由 `GroupNorm` 替换为 `SpatialNorm`。

### 4.4 MOVQDecoder 整体前向

```
输入 z:  [B, z_channels, h, w]
输入 zq: [B, embed_dim,  h, w]   ← 量化特征，作为条件信号

  │
  ▼
conv_in: Conv2d(z_channels, block_in, 3×3, pad=1)   → [B, block_in, h, w]
  │
  ▼
中间层（bottleneck）
  mid.block_1: ResnetBlock(block_in, block_in, zq)
  mid.attn_1:  AttnBlock(block_in, zq)
  mid.block_2: ResnetBlock(block_in, block_in, zq)
  │
  ▼
上采样阶段（num_resolutions 层，从高层到低层）
  每层：
    (num_res_blocks+1) × ResnetBlock(block_in→block_out, zq)
    （若当前分辨率在 attn_resolutions 中）× AttnBlock(block_out, zq)
    Upsample（scale×2）→ 分辨率翻倍
  │
  ▼
norm_out: SpatialNorm(block_in, zq)
swish
conv_out: Conv2d(block_in, out_ch, 3×3, pad=1)
  │
  ▼
输出: [B, out_ch, H, W]   （out_ch=3 对应 RGB 图像）
```

$zq$ 在每一个 `SpatialNorm` 调用中都会被插值到当前特征图的空间尺寸，因此它在整个解码过程中持续提供位置感知的条件信息。

---

## 5. MOVQ 顶层前向传播

> 文件：`movqgan/models/vqgan.py`，类 `MOVQ`

### 5.1 编码路径

```
输入 x: [B, 3, H, W]
  │
  ▼
encoder(x)                                    → [B, z_channels, h, w]
  │
  ▼
quant_conv: Conv2d(z_channels, embed_dim, 1)  → [B, embed_dim, h, w]
  │
  ▼
quantize(h)                                   → quant [B, embed_dim, h, w]
                                                 emb_loss (scalar)
                                                 indices [B, h, w]
```

`quant_conv` 为 $1\times1$ 卷积，等价于对每个空间位置做线性投影：$\mathbb{R}^{z\_channels} \to \mathbb{R}^{embed\_dim}$。

### 5.2 解码路径

```
输入 quant: [B, embed_dim, h, w]
  │
  ├─── post_quant_conv: Conv2d(embed_dim, z_channels, 1)  → quant2 [B, z_channels, h, w]
  │
  ▼
decoder(quant2, quant)
         ↑         ↑
         z         zq（原始量化特征，作为条件信号）
  │
  ▼
输出 dec: [B, 3, H, W]
```

`post_quant_conv` 同样为 $1\times1$ 卷积：$\mathbb{R}^{embed\_dim} \to \mathbb{R}^{z\_channels}$。

### 5.3 完整前向（推理时）

$$\begin{aligned}
h &= \text{Encoder}(x) \\
h &= \text{quant\_conv}(h) \\
(z_q, \mathcal{L}_{VQ}, k^*) &= \text{VectorQuantizer}(h) \\
z_q' &= \text{post\_quant\_conv}(z_q) \\
\hat{x} &= \text{MOVQDecoder}(z_q',\ z_q)
\end{aligned}$$

其中 $z_q$ 同时扮演两个角色：
1. 经 `post_quant_conv` 映射后作为解码器的主特征输入
2. 直接作为 $zq$ 条件信号传入解码器的每个 `SpatialNorm`

---

## 6. MoVQGAN 与 VQGAN 的核心区别

### 6.1 解码器归一化方式

| 模块 | VQGAN | MoVQGAN |
|---|---|---|
| 归一化层 | `GroupNorm`（固定仿射参数） | `SpatialNorm`（由 $zq$ 动态生成 $\gamma, \beta$） |
| 条件信息 | 无 | 量化特征 $zq$ |
| 空间自适应 | 否 | 是 |

### 6.2 解码器输入

```python
# VQGAN（VQModel）
def decode(self, quant):
    quant = self.post_quant_conv(quant)
    dec = self.decoder(quant)          # 单输入
    return dec

# MoVQGAN（MOVQ）
def decode(self, quant):
    quant2 = self.post_quant_conv(quant)
    dec = self.decoder(quant2, quant)  # 双输入：主特征 + zq 条件
    return dec
```

### 6.3 编码器

两者编码器结构完全相同，均使用标准 `GroupNorm`，无条件信号。

### 6.4 SpatialNorm 的数学含义

标准 GroupNorm 的仿射参数 $\gamma_c, \beta_c$ 是全局共享的标量，对所有空间位置施加相同的缩放与偏移。

SpatialNorm 将其替换为位置相关的映射：

$$\gamma(i,j) = f_\gamma(zq_{i,j}), \quad \beta(i,j) = f_\beta(zq_{i,j})$$

其中 $f_\gamma, f_\beta$ 为 $1\times1$ 卷积（线性投影）。这使得解码器在每个空间位置的特征调制都受到对应位置量化码的影响，从而在解码过程中保留了更多来自码本的结构信息。

### 6.5 差异汇总

| 维度 | VQGAN | MoVQGAN |
|---|---|---|
| 解码器归一化 | GroupNorm | SpatialNorm（zq 条件） |
| 解码器输入数量 | 1（`quant2`） | 2（`quant2` + `zq`） |
| 编码器 | 标准 GroupNorm | 标准 GroupNorm（相同） |
| 量化器 | VectorQuantizer2 | VectorQuantizer2（相同） |
| 条件信号传播 | 无 | $zq$ 贯穿解码器所有层 |
| 空间自适应调制 | 否 | 是（每位置独立 $\gamma, \beta$） |
