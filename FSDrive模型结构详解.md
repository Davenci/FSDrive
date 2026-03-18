# FSDrive模型结构详解

## 1. 整体架构

FSDrive是一个端到端的自动驾驶视觉语言模型，通过视觉思维链（Spatio-Temporal CoT）实现轨迹规划。整体架构包括三个主要部分：

```
数据处理阶段：
nuScenes图像 → MoVQGAN Encoder → 视觉Tokens (离散化)

训练/推理阶段：
6个相机图像 + 文本描述 → Qwen2-VL-2B-Instruct → 视觉Tokens + 轨迹Waypoints

可视化阶段：
视觉Tokens → MoVQGAN Decoder → 未来图像
```

### 1.1 数据流

**输入**：
- 当前时刻的6个相机图像（CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT）
- 原始分辨率：1600×900
- 场景描述文本（包括车辆状态、周围物体信息等）

**中间表示**：
- 图像预处理：Resize到128×192，归一化到[-1, 1]
- MoVQGAN编码：图像 → 视觉Tokens（384个离散tokens，形状为16×24）
- Tokens范围：0-8191（codebook大小为8192）

**输出**：
- 未来0.5秒的CAM_FRONT图像的视觉Tokens（384个）
- 未来3秒的轨迹Waypoints（6个点，每0.5秒一个）
- Waypoints格式：(x, y)坐标，单位为米

---

## 2. MoVQGAN模型结构

MoVQGAN是一个基于向量量化的视频生成模型，用于将图像编码为离散的视觉tokens。

### 2.1 整体结构

MoVQGAN模型包含以下主要组件：

```
输入图像 (B, 3, 128, 192)
    ↓
Encoder (下采样网络)
    ↓
quant_conv (1x1卷积, z_channels → embed_dim)
    ↓
VectorQuantizer (向量量化, codebook大小=8192)
    ↓
post_quant_conv (1x1卷积, embed_dim → z_channels)
    ↓
Decoder (上采样网络, 使用SpatialNorm)
    ↓
输出图像 (B, 3, 128, 192)
```

### 2.2 Encoder详细结构

Encoder是一个下采样网络，将输入图像从(B, 3, 128, 192)编码为(B, z_channels, 16, 24)的特征图。

**参数配置**（以270M模型为例）：
- ch: 128（基础通道数）
- ch_mult: (1, 2, 4)（通道倍数）
- num_res_blocks: 2（每层的ResnetBlock数量）
- attn_resolutions: []（注意力层的分辨率）
- z_channels: 4（输出通道数）
- resolution: 128（输入分辨率的高度）
- in_channels: 3（输入通道数，RGB）

**网络结构**：

1. **conv_in**：
   - Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
   - 输出：(B, 128, 128, 192)

2. **down[0]**（第一层下采样）：
   - ResnetBlock(128, 128) × 2
   - Downsample(128)（stride=2）
   - 输出：(B, 128, 64, 96)

3. **down[1]**（第二层下采样）：
   - ResnetBlock(128, 256) × 2
   - Downsample(256)（stride=2）
   - 输出：(B, 256, 32, 48)

4. **down[2]**（第三层下采样）：
   - ResnetBlock(256, 512) × 2
   - 无Downsample（最后一层）
   - 输出：(B, 512, 32, 48)

5. **mid**（中间层）：
   - ResnetBlock(512, 512)
   - AttnBlock(512)
   - ResnetBlock(512, 512)
   - 输出：(B, 512, 32, 48)

6. **conv_out**：
   - GroupNorm(512)
   - Swish激活
   - Conv2d(512, 8, kernel_size=3, stride=1, padding=1)（double_z=True，所以输出2*z_channels=8）
   - 输出：(B, 8, 32, 48)

**注意**：实际输出的特征图大小为(B, 8, 32, 48)，但经过quant_conv后会变为(B, 4, 32, 48)，然后经过VectorQuantizer后变为(B, 4, 16, 24)。

### 2.3 Decoder详细结构（MOVQDecoder）

Decoder是一个上采样网络，将量化后的特征图从(B, 4, 16, 24)解码为(B, 3, 128, 192)的图像。

**关键特性**：
- 使用SpatialNorm代替普通的GroupNorm
- SpatialNorm利用量化特征zq来调制解码器的特征

**参数配置**（以270M模型为例）：
- ch: 128（基础通道数）
- ch_mult: (1, 2, 4)（通道倍数）
- num_res_blocks: 2（每层的ResnetBlock数量）
- attn_resolutions: []（注意力层的分辨率）
- z_channels: 4（输入通道数）
- out_ch: 3（输出通道数，RGB）

**网络结构**：

1. **conv_in**：
   - Conv2d(4, 512, kernel_size=3, stride=1, padding=1)
   - 输入：(B, 4, 16, 24)
   - 输出：(B, 512, 16, 24)

2. **mid**（中间层）：
   - ResnetBlock(512, 512, 使用SpatialNorm)
   - AttnBlock(512)
   - ResnetBlock(512, 512, 使用SpatialNorm)
   - 输出：(B, 512, 16, 24)

3. **up[0]**（第一层上采样）：
   - ResnetBlock(512, 512, 使用SpatialNorm) × 3
   - Upsample(512)（scale_factor=2）
   - 输出：(B, 512, 32, 48)

4. **up[1]**（第二层上采样）：
   - ResnetBlock(512, 256, 使用SpatialNorm) × 3
   - Upsample(256)（scale_factor=2）
   - 输出：(B, 256, 64, 96)

5. **up[2]**（第三层上采样）：
   - ResnetBlock(256, 128, 使用SpatialNorm) × 3
   - Upsample(128)（scale_factor=2）
   - 输出：(B, 128, 128, 192)

6. **conv_out**：
   - SpatialNorm(128)
   - Swish激活
   - Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
   - 输出：(B, 3, 128, 192)

**SpatialNorm的作用**：
- 传统的GroupNorm：`norm_f = GroupNorm(f)`
- SpatialNorm：`new_f = norm_f * conv_y(zq) + conv_b(zq)`
- 其中zq是量化后的特征，通过插值到与f相同的空间分辨率
- conv_y和conv_b是1x1卷积，用于生成调制参数

### 2.4 VectorQuantizer详细结构

VectorQuantizer将连续的特征向量映射到离散的codebook中，实现特征的量化。

**参数配置**：
- n_e: 8192（codebook大小，即可用的token数量）
- e_dim: 4（embedding维度）
- beta: 0.25（commitment loss的权重）

**核心组件**：
- embedding: nn.Embedding(8192, 4)（codebook）

**前向传播流程**：

1. **输入重排**：
   - 输入：z (B, 4, 32, 48)
   - 重排：z = rearrange(z, 'b c h w -> b h w c')
   - 展平：z_flattened = z.view(-1, 4)  # (B*32*48, 4)

2. **计算距离**：
   - 计算输入与codebook中每个向量的欧氏距离
   - d = ||z_flattened||² + ||embedding.weight||² - 2 * z_flattened @ embedding.weight.T
   - d的形状：(B*32*48, 8192)

3. **选择最近的code**：
   - min_encoding_indices = argmin(d, dim=1)  # (B*32*48,)
   - z_q = embedding(min_encoding_indices).view(z.shape)  # (B, 32, 48, 4)

4. **计算loss**：
   - commitment_loss = ||z_q.detach() - z||²（编码器承诺损失）
   - codebook_loss = ||z_q - z.detach()||²（codebook更新损失）
   - total_loss = commitment_loss + beta * codebook_loss

5. **Straight-Through Estimator**：
   - z_q = z + (z_q - z).detach()
   - 前向传播使用量化后的z_q，反向传播梯度直接传递给z

6. **输出重排**：
   - z_q = rearrange(z_q, 'b h w c -> b c h w')  # (B, 4, 32, 48)
   - 返回：z_q, loss, min_encoding_indices

**输出**：
- z_q: 量化后的特征 (B, 4, 32, 48)
- loss: 量化损失（标量）
- min_encoding_indices: 选择的token索引 (B, 32, 48)，范围0-8191

**注意**：在FSDrive中，实际使用的indices形状为(B, 16, 24)，共384个tokens。这是因为在quant_conv之后还有额外的下采样操作。

### 2.5 AttnBlock详细结构

AttnBlock实现了自注意力机制，用于捕捉特征图中的长距离依赖关系。

**核心组件**：
- norm: GroupNorm(num_groups=32, num_channels=in_channels)
- q: Conv2d(in_channels, in_channels, kernel_size=1)（Query投影）
- k: Conv2d(in_channels, in_channels, kernel_size=1)（Key投影）
- v: Conv2d(in_channels, in_channels, kernel_size=1)（Value投影）
- proj_out: Conv2d(in_channels, in_channels, kernel_size=1)（输出投影）

**前向传播流程**：

1. **归一化**：
   - h_ = norm(x)  # (B, C, H, W)

2. **计算Q, K, V**：
   - q = q(h_)  # (B, C, H, W)
   - k = k(h_)  # (B, C, H, W)
   - v = v(h_)  # (B, C, H, W)

3. **重排Q, K, V**：
   - q = q.reshape(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
   - k = k.reshape(B, C, H*W)  # (B, C, H*W)
   - v = v.reshape(B, C, H*W)  # (B, C, H*W)

4. **计算注意力权重**：
   - w_ = q @ k  # (B, H*W, H*W)
   - w_ = w_ * (C ** -0.5)  # 缩放
   - w_ = softmax(w_, dim=2)  # (B, H*W, H*W)

5. **应用注意力**：
   - w_ = w_.permute(0, 2, 1)  # (B, H*W, H*W)
   - h_ = v @ w_  # (B, C, H*W)
   - h_ = h_.reshape(B, C, H, W)  # (B, C, H, W)

6. **输出投影和残差连接**：
   - h_ = proj_out(h_)  # (B, C, H, W)
   - return x + h_  # 残差连接

**计算复杂度**：
- 注意力矩阵的大小：(H*W, H*W)
- 对于32×48的特征图，注意力矩阵大小为(1536, 1536)
- 计算复杂度：O((H*W)² * C)

### 2.6 ResnetBlock详细结构

ResnetBlock是一个残差块，包含两个卷积层和残差连接。

**核心组件**：
- norm1: GroupNorm(num_groups=32, num_channels=in_channels)
- conv1: Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
- norm2: GroupNorm(num_groups=32, num_channels=out_channels)
- dropout: Dropout(dropout)
- conv2: Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
- shortcut: Conv2d(in_channels, out_channels, kernel_size=1)（如果in_channels != out_channels）

**前向传播流程**：

1. **第一个卷积块**：
   - h = norm1(x)
   - h = swish(h)  # swish(x) = x * sigmoid(x)
   - h = conv1(h)

2. **第二个卷积块**：
   - h = norm2(h)
   - h = swish(h)
   - h = dropout(h)
   - h = conv2(h)

3. **残差连接**：
   - if in_channels != out_channels:
       - x = shortcut(x)
   - return x + h

**在MOVQDecoder中的特殊之处**：
- 使用SpatialNorm代替GroupNorm
- SpatialNorm需要额外的量化特征zq作为输入
- 前向传播：h = norm1(x, zq)，其中zq用于调制归一化后的特征

---

## 3. Loss函数

FSDrive的训练包含两个阶段，每个阶段使用不同的loss函数。

### 3.1 MoVQGAN的Loss函数（VQLPIPSWithDiscriminator2）

MoVQGAN使用对抗训练，包含生成器（自编码器）和判别器两部分。

**生成器Loss（optimizer_idx=0）**：

1. **重建损失（Reconstruction Loss）**：
   - rec_loss = (inputs - reconstructions)²
   - 逐像素的L2损失
   - 形状：(B, 3, 128, 192)

2. **感知损失（Perceptual Loss）**：
   - p_loss = LPIPS(inputs, reconstructions)
   - 使用预训练的VGG网络提取特征
   - 计算特征空间的距离
   - 权重：0.1

3. **Codebook损失（Codebook Loss）**：
   - codebook_loss = ||z_q.detach() - z||² + 0.25 * ||z_q - z.detach()||²
   - 来自VectorQuantizer
   - 权重：1.0

4. **生成器对抗损失（Generator Adversarial Loss）**：
   - logits_fake = discriminator(reconstructions)
   - g_loss = -mean(logits_fake)
   - 权重：0.1

**生成器总损失**：
```
total_loss = rec_loss + 0.1 * p_loss + 0.1 * g_loss + 1.0 * codebook_loss
```

**判别器Loss（optimizer_idx=1）**：

1. **Hinge Loss**：
   - logits_real = discriminator(inputs)
   - logits_fake = discriminator(reconstructions.detach())
   - loss_real = mean(relu(1 - logits_real))
   - loss_fake = mean(relu(1 + logits_fake))
   - d_loss = 0.5 * (loss_real + loss_fake)

**判别器结构**：
- NLayerDiscriminator
- 3层卷积网络
- 输入：(B, 3, 128, 192)
- 输出：(B, 1, H', W')（每个位置的真假判断）

### 3.2 大模型的Loss函数

大模型（Qwen2-VL-2B-Instruct）使用标准的语言模型loss。

**交叉熵损失（Cross-Entropy Loss）**：

1. **输入**：
   - 6个相机图像（作为<image>标记）
   - 文本描述（场景信息）

2. **目标**：
   - 视觉tokens：<|0|>, <|1|>, ..., <|8191|>（384个tokens）
   - 轨迹waypoints：文本格式的坐标

3. **Loss计算**：
   - 对于每个token位置，计算预测分布与真实token的交叉熵
   - loss = -log P(token_i | context)
   - 只计算assistant部分的loss（user部分的loss被mask掉）

**训练配置**：
- learning_rate: 1.0e-4
- num_train_epochs: 16
- freeze_vision_tower: true（冻结视觉编码器）
- freeze_multi_modal_projector: true（冻结多模态投影器）
- 只训练语言模型部分

---

## 4. 训练和推理流程

### 4.1 数据准备流程

**步骤1：提取视觉tokens**

使用MoVQGAN模型将图像编码为离散的视觉tokens：

```python
# 预处理图像
img = Image.open(img_path)
img = resize(img, (128, 192))
img = (img / 127.5) - 1  # 归一化到[-1, 1]

# 编码为tokens
with torch.no_grad():
    indices = movqgan_model(img)  # (1, 16, 24)
```

**步骤2：构建训练数据**

将视觉tokens与轨迹数据结合，构建对话格式的数据：

```json
{
    "id": "sample_token",
    "images": [
        "path/to/CAM_FRONT.jpg",
        "path/to/CAM_FRONT_LEFT.jpg",
        ...
    ],
    "system": "You're an autonomous vehicle's brain...",
    "conversations": [
        {
            "from": "human",
            "value": "Here are current six images... [场景描述]"
        },
        {
            "from": "gpt",
            "value": "<|token1|><|token2|>...<|token384|> These are the visual tokens... [轨迹waypoints]"
        }
    ]
}
```

### 4.2 训练流程

**阶段1：预训练（Pretrain）**

目标：激活VLM的视觉生成能力

```bash
cd LLaMA-Factory
llamafactory-cli train ../configs/pretrain.yaml
```

配置：
- 数据集：大量的图像-tokens对
- 任务：给定当前图像，预测未来图像的tokens
- 训练轮数：根据数据量调整

**阶段2：微调（SFT）**

目标：让VLM学会视觉思考轨迹规划

```bash
llamafactory-cli train ../configs/sft.yaml
```

配置：
- 数据集：6个相机图像 + 场景描述 → 未来图像tokens + 轨迹waypoints
- 任务：给定当前场景，预测未来图像tokens和轨迹
- 训练轮数：16
- 学习率：1.0e-4

### 4.3 推理流程

**步骤1：模型推理**

使用训练好的模型进行推理：

```bash
python scripts/vllm_infer.py \
--model_name_or_path saves/qwen2_vl-2b/sft \
--dataset val_cot_motion \
--template qwen2_vl \
--cutoff_len 32768 \
--max_new_tokens 2048 \
--temperature 0.1 \
--top_p 0.1 \
--top_k 10
```

输出：results.jsonl（包含预测的tokens和轨迹）

**步骤2：解析结果**

从模型输出中提取视觉tokens和轨迹waypoints：

```python
# 解析视觉tokens
tokens = extract_tokens(output)  # 提取<|token|>格式的tokens
tokens = [int(t) for t in tokens]  # 转换为整数

# 解析轨迹waypoints
waypoints = extract_waypoints(output)  # 提取坐标
```

**步骤3：可视化CoT**

使用MoVQGAN解码器将视觉tokens还原为图像：

```bash
python ./MoVQGAN/vis.py \
--input_json ./LLaMA-Factory/eval_traj.json \
--output_dir ./vis_cot
```

解码流程：
```python
# tokens → embedding
quant = movqgan_model.quantize.embedding(tokens)  # (384, 4)
quant = quant.view(1, 16, 24, 4)
quant = rearrange(quant, 'b h w c -> b c h w')  # (1, 4, 16, 24)

# embedding → image
quant2 = movqgan_model.post_quant_conv(quant)
img = movqgan_model.decoder(quant2, quant)  # (1, 3, 128, 192)
```

### 4.4 评估流程

**步骤1：匹配预测结果**

```bash
python tools/match.py \
--pred_trajs_path ./LLaMA-Factory/results.jsonl \
--token_traj_path ./LLaMA-Factory/data/val_cot_motion.json
```

**步骤2：计算评估指标**

```bash
python tools/evaluation/evaluation.py \
--metric uniad \
--result_file ./LLaMA-Factory/eval_traj.json
```

评估指标：
- L2距离：预测轨迹与真实轨迹的欧氏距离
- 碰撞率：预测轨迹与障碍物的碰撞概率

---

## 5. 总结

FSDrive通过以下创新实现了端到端的自动驾驶轨迹规划：

1. **视觉思维链（Spatio-Temporal CoT）**：
   - 使用MoVQGAN将图像编码为离散的视觉tokens
   - 大模型预测未来图像的tokens，实现视觉思考
   - 将视觉tokens与轨迹规划统一在一个框架中

2. **模型架构**：
   - MoVQGAN：基于向量量化的视频生成模型
   - Qwen2-VL-2B-Instruct：视觉语言大模型
   - 冻结视觉编码器，只训练语言模型部分

3. **训练策略**：
   - 预训练：激活视觉生成能力
   - 微调：学会视觉思考轨迹规划
   - 使用对抗训练提高图像质量

4. **关键技术**：
   - VectorQuantizer：将连续特征映射到离散codebook
   - SpatialNorm：利用量化特征调制解码器
   - AttnBlock：捕捉长距离依赖关系
   - Straight-Through Estimator：解决离散化的梯度问题

这个架构首次实现了自动驾驶中的视觉推理，为端到端自动驾驶提供了新的思路。
