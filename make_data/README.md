# 数据集制作脚本使用说明

## 概述

`build_dataset.py` 是一个三阶段的异常检测微调数据集制作脚本，输入为原始图片，输出为 LLaMA-Factory 可用的 ShareGPT 格式 JSON。

## 目录结构要求

`data/` 目录下按如下结构组织图片：

``` 
make_data/data/
└── <类别名>/
    ├── nomal/       # 正常样本图片（.png / .jpg）
    └── anomal/      # 异常样本图片（.png / .jpg）
```

示例：

```
make_data/data/
└── losiding/
    ├── nomal/
    │   ├── 001.png
    │   └── 002.png
    └── anomal/
        ├── 001.png
        └── 002.png
```

支持同时放置多个类别目录，脚本会自动遍历所有类别。

## 三阶段流程

| 阶段 | 内容 | 输出 |
|------|------|------|
| 阶段一 | 对 `nomal/` 图片用 MoVQGAN 提取 384-token 序列 | `MoVQGAN/gt_indices_{类别}_normal.json` |
| 阶段二 | 对 `anomal/` 图片做随机增强（旋转/亮度/对比度/裁剪），补足至目标数量 | `LLaMA-Factory/data/{类别}_augmented/` |
| 阶段三 | 将阶段一、二的结果组合为 ShareGPT 格式数据集 | `LLaMA-Factory/data/{类别}_anomaly_sft.json` |

## 快速开始

```bash
cd FSDrive
python make_data/build_dataset.py
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--category` | 无（处理全部） | 指定单个类别目录名，如 `losiding` |
| `--aug_count` | `200` | 每个类别异常样本增强后的目标总数 |
| `--device` | `cuda` | 推理设备，无 GPU 时设为 `cpu` |
| `--skip_stage1` | `False` | 跳过阶段一，使用已有的 `gt_indices_normal.json` |

## 使用示例

**处理单个类别：**

```bash
python make_data/build_dataset.py --category losiding --aug_count 600
```

**处理全部类别：**

```bash
python make_data/build_dataset.py
```

**指定类别 + 跳过阶段一（已有 Token 文件）：**

```bash
python make_data/build_dataset.py --category losiding --skip_stage1
```

**无 GPU 环境：**

```bash
python make_data/build_dataset.py --category losiding --device cpu
```

## 输出格式

`losiding_anomaly_sft.json` 中每条数据的格式如下：

```json
{
  "id": "losiding_6c2fa9ee0f844",
  "images": ["losiding_augmented/losiding/losiding_aug_0000_001.png"],
  "system": "你是异常检测专家，负责产品质量鉴定。",
  "conversations": [
    {
      "from": "human",
      "value": "你眼前的是一个losiding<image>\n，但该产品存在异常。请结合你对正常产品的认知进行对比，并还原该产品理想状态下的样子。最后请判断该losiding是否有异常，并指出异常位置。"
    },
    {
      "from": "gpt",
      "value": "<|196|><|8641|>...<|im_end|>"
    }
  ]
}
```

## 注意事项

- 阶段一依赖 MoVQGAN 模型权重，需提前确认 `MoVQGAN/` 目录下的 checkpoint 已就位。
- `nomal/` 目录名为固定拼写（非 `normal`），与原始数据集保持一致。
- 若某类别下 `anomal/` 原始图片数量已超过 `--aug_count`，脚本不会截断，实际数量以原始数为准。
