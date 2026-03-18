"""
果汁异常检测数据集一键制作脚本
对应 dataset_task.md 三阶段流程：
  阶段一：正常样本 MoVQGAN Token 提取
  阶段二：异常样本图像增强
  阶段三：ShareGPT 格式微调数据集制作
"""

import os
import sys
import json
import uuid
import random
import shutil
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ──────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # FSDrive/
JUICE_ROOT = Path(__file__).resolve().parent / "juice" # make_data/juice/
LLAMA_DATA = ROOT / "LLaMA-Factory" / "data"
MOVQGAN_DIR = ROOT / "MoVQGAN"

NORMAL_TOKEN_JSON  = MOVQGAN_DIR / "gt_indices_juice_normal.json"
AUGMENTED_IMG_DIR  = LLAMA_DATA / "juice_augmented"
OUTPUT_DATASET_JSON = LLAMA_DATA / "juice_anomaly_sft.json"

SYSTEM_PROMPT = "你是异常检测专家，负责产品质量鉴定。"
HUMAN_PROMPT  = ("你眼前的是一瓶果汁，但该产品存在异常。"
                 "请结合你对正常产品的认知进行对比，并还原该产品理想状态下的样子。"
                 "最后请判断该果汁是否有异常，并指出异常位置。")

# ──────────────────────────────────────────────────────────
# MoVQGAN 图像预处理（与 sft_data.py 保持一致）
# ──────────────────────────────────────────────────────────
def prepare_image(img: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((128, 192), interpolation=T.InterpolationMode.BICUBIC),
    ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB")).astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))


# ──────────────────────────────────────────────────────────
# 阶段一：正常样本 Token 提取
# ──────────────────────────────────────────────────────────
def stage1_extract_tokens(device: str = "cuda") -> dict:
    """
    遍历 juice/<category>/nomal/ 下所有图片，用 MoVQGAN 提取 384-token 序列。
    返回 {"category": {"img_id": [t1, t2, ...]}, ...}
    同时保存到 NORMAL_TOKEN_JSON。
    """
    # 延迟导入，只在执行阶段一时才加载模型
    sys.path.insert(0, str(MOVQGAN_DIR))
    from movqgan import get_movqgan_model  # noqa: E402

    print("=== 阶段一：加载 MoVQGAN 模型 ===")
    # get_movqgan_model 内部 hardcode 了 checkpoint 路径为相对路径，需切换 cwd
    orig_cwd = os.getcwd()
    os.chdir(str(ROOT))
    model = get_movqgan_model("270M", pretrained=True, device=device)
    os.chdir(orig_cwd)

    all_tokens: dict = {}   # category -> {img_id: [tokens]}

    categories = [d.name for d in JUICE_ROOT.iterdir() if d.is_dir()]
    for category in sorted(categories):
        normal_dir = JUICE_ROOT / category / "nomal"
        if not normal_dir.exists():
            print(f"  [跳过] {category} 无 nomal 目录")
            continue

        img_files = sorted(normal_dir.glob("*.png")) + sorted(normal_dir.glob("*.jpg"))
        cat_tokens: dict = {}

        print(f"  处理类别 [{category}]，共 {len(img_files)} 张正常图片")
        for img_path in tqdm(img_files, desc=f"  {category}"):
            # 用 类别_哈希 作唯一ID
            stem = img_path.stem
            img_id = f"{category}_{stem}"

            img = Image.open(img_path)
            tensor = prepare_image(img).to(device).unsqueeze(0)

            with torch.no_grad():
                indices = model(tensor)  # forward() 返回 indices

            tokens: list = indices.cpu().flatten().tolist()
            assert len(tokens) == 384, f"期望384 tokens，得到 {len(tokens)}，请检查图片分辨率"
            cat_tokens[img_id] = tokens

        all_tokens[category] = cat_tokens
        print(f"  [{category}] 完成，共 {len(cat_tokens)} 条 token 序列")

    # 按任务要求保存为扁平格式 {"类别_图片ID": [tokens]}，同时保留分类索引供阶段三使用
    flat_tokens = {}
    for cat, items in all_tokens.items():
        for img_id, tokens in items.items():
            flat_tokens[img_id] = tokens

    NORMAL_TOKEN_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(NORMAL_TOKEN_JSON, "w") as f:
        json.dump(flat_tokens, f, indent=4)
    print(f"[阶段一] Token 映射已保存至 {NORMAL_TOKEN_JSON}，共 {len(flat_tokens)} 条\n")
    return all_tokens  # 内部仍返回嵌套结构供阶段三按类别抽取


# ──────────────────────────────────────────────────────────
# 阶段二：异常样本图像增强
# ──────────────────────────────────────────────────────────
def augment_image(img: Image.Image) -> Image.Image:
    """随机旋转 / 亮度 / 对比度 / 轻微裁剪。"""
    # 随机旋转 ±15°
    angle = random.uniform(-15, 15)
    img = TF.rotate(img, angle)

    # 随机亮度
    brightness_factor = random.uniform(0.7, 1.3)
    img = TF.adjust_brightness(img, brightness_factor)

    # 随机对比度
    contrast_factor = random.uniform(0.7, 1.3)
    img = TF.adjust_contrast(img, contrast_factor)

    # 轻微随机裁剪（保留 85%-100%）
    w, h = img.size
    scale = random.uniform(0.85, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    left = random.randint(0, w - new_w)
    top  = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((w, h), Image.BICUBIC)

    return img


def stage2_augment_anomalies(aug_count: int = 200) -> dict:
    """
    对每个类别的异常图片做增强，使每类总数达到 aug_count。
    增强后图片保存到 AUGMENTED_IMG_DIR/<category>/
    返回 {"category": [相对于 LLaMA-Factory/data/ 的路径, ...], ...}
    Category-Lock：保证增强图片所属类别与原始类别完全一致。
    """
    print("=== 阶段二：异常样本图像增强 ===")
    AUGMENTED_IMG_DIR.mkdir(parents=True, exist_ok=True)

    augmented_paths: dict = {}  # category -> [relative_path, ...]

    categories = [d.name for d in JUICE_ROOT.iterdir() if d.is_dir()]
    for category in sorted(categories):
        anomal_dir = JUICE_ROOT / category / "anomal"
        if not anomal_dir.exists():
            print(f"  [跳过] {category} 无 anomal 目录")
            continue

        out_dir = AUGMENTED_IMG_DIR / category
        out_dir.mkdir(parents=True, exist_ok=True)

        src_files = sorted(anomal_dir.glob("*.png")) + sorted(anomal_dir.glob("*.jpg"))
        if not src_files:
            print(f"  [跳过] {category}/anomal 无图片")
            continue

        cat_paths: list = []
        existing = len(src_files)
        print(f"  [{category}] 原始异常图 {existing} 张 → 目标 {aug_count} 张")

        # 先把原始图复制过来（保持 Category-Lock 标签）
        for src in src_files:
            dst = out_dir / f"{category}_orig_{src.stem}.png"
            shutil.copy2(src, dst)
            # 路径相对于 LLAMA_DATA
            cat_paths.append(str(dst.relative_to(LLAMA_DATA)))

        # 增强补足到 aug_count
        need = aug_count - len(cat_paths)
        aug_idx = 0
        while need > 0:
            src = random.choice(src_files)
            img = Image.open(src).convert("RGB")
            aug = augment_image(img)
            aug_name = f"{category}_aug_{aug_idx:04d}_{src.stem}.png"
            dst = out_dir / aug_name
            aug.save(dst)
            cat_paths.append(str(dst.relative_to(LLAMA_DATA)))
            aug_idx += 1
            need -= 1

        augmented_paths[category] = cat_paths
        print(f"  [{category}] 完成，共 {len(cat_paths)} 张")

    print(f"[阶段二] 增强图片已保存至 {AUGMENTED_IMG_DIR}\n")
    return augmented_paths


# ──────────────────────────────────────────────────────────
# 阶段三：ShareGPT 格式数据集制作
# ──────────────────────────────────────────────────────────
def tokens_to_gpt_value(tokens: list) -> str:
    """将 token 列表编码为 <|ID|> 格式字符串，附加说明文字。"""
    token_str = "".join(f"<|{t}|>" for t in tokens)
    suffix = (" \n以上是该类果汁正常状态的特征映射。"
              "对比可见，当前产品存在异常。\n"
              "<|endoftext|><|im_end|>")
    return token_str + suffix


def stage3_build_sharegpt(all_tokens: dict, augmented_paths: dict) -> list:
    """
    Category-Lock：异常图片只配对同类别的正常 Token。
    返回 ShareGPT 格式 list，同时写入 OUTPUT_DATASET_JSON。
    """
    print("=== 阶段三：构建 ShareGPT 格式数据集 ===")
    dataset: list = []

    for category, img_paths in augmented_paths.items():
        if category not in all_tokens:
            print(f"  [跳过] {category} 无对应正常 Token（Category-Lock 校验失败）")
            continue

        cat_token_items = list(all_tokens[category].items())  # [(img_id, tokens), ...]
        if not cat_token_items:
            print(f"  [跳过] {category} Token 库为空")
            continue

        for img_rel_path in tqdm(img_paths, desc=f"  {category}"):
            # Category-Lock：从同类 Token 库随机抽取
            _, tokens = random.choice(cat_token_items)

            sample_id = f"{category}_{uuid.uuid4().hex[:13]}"
            gpt_value = tokens_to_gpt_value(tokens)

            record = {
                "id": sample_id,
                "images": [img_rel_path],
                "system": SYSTEM_PROMPT,
                "conversations": [
                    {"from": "human", "value": HUMAN_PROMPT},
                    {"from": "gpt",   "value": gpt_value},
                ],
            }
            dataset.append(record)

        print(f"  [{category}] 生成 {len(img_paths)} 条训练样本")

    OUTPUT_DATASET_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATASET_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"[阶段三] 数据集已保存至 {OUTPUT_DATASET_JSON}，共 {len(dataset)} 条\n")
    return dataset


# ──────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="果汁异常检测数据集一键制作")
    parser.add_argument("--aug_count", type=int, default=200,
                        help="每类异常样本增强后目标总数（默认200）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch 设备（默认 cuda，无 GPU 可设为 cpu）")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="跳过阶段一（使用已有的 gt_indices_juice_normal.json）")
    args = parser.parse_args()

    random.seed(42)

    # ── 阶段一 ──
    if args.skip_stage1:
        if not NORMAL_TOKEN_JSON.exists():
            raise FileNotFoundError(f"--skip_stage1 但未找到 {NORMAL_TOKEN_JSON}")
        print(f"[阶段一] 跳过，加载已有 Token 文件: {NORMAL_TOKEN_JSON}")
        with open(NORMAL_TOKEN_JSON) as f:
            loaded = json.load(f)
        # 自动兼容扁平格式和旧嵌套格式
        first_val = next(iter(loaded.values()))
        if isinstance(first_val, list):
            # 扁平格式 {"cat_imgid": [tokens]} → 嵌套
            all_tokens: dict = {}
            for img_id, tokens in loaded.items():
                cat = img_id.split("_")[0]
                all_tokens.setdefault(cat, {})[img_id] = tokens
        else:
            # 已经是嵌套格式 {"cat": {"img_id": [tokens]}}
            all_tokens = loaded
    else:
        all_tokens = stage1_extract_tokens(device=args.device)

    # ── 阶段二 ──
    augmented_paths = stage2_augment_anomalies(aug_count=args.aug_count)

    # ── 阶段三 ──
    stage3_build_sharegpt(all_tokens, augmented_paths)

    print("=== 全流程完成 ===")
    print(f"  正常 Token 文件 : {NORMAL_TOKEN_JSON}")
    print(f"  增强异常图片目录 : {AUGMENTED_IMG_DIR}")
    print(f"  微调数据集      : {OUTPUT_DATASET_JSON}")


if __name__ == "__main__":
    main()
