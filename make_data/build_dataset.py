"""
异常检测数据集一键制作脚本
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
from multiprocessing import Pool, cpu_count

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
JUICE_ROOT = Path(__file__).resolve().parent / "data"  # make_data/data/
LLAMA_DATA = ROOT / "LLaMA-Factory" / "data"
MOVQGAN_DIR = ROOT / "MoVQGAN"

# Token JSON / 输出数据集路径均按类别动态生成，见各阶段函数

SYSTEM_PROMPT = "你是异常检测专家，负责产品质量鉴定。"


def make_human_prompt(category: str) -> str:
    return (f"你眼前的是一个{category}<image>\n，但该产品存在异常。"
            f"请结合你对正常产品的认知进行对比，并还原该产品理想状态下的样子。"
            f"最后请判断该{category}是否有异常，并指出异常位置。")


def tokens_to_gpt_value(tokens: list, category: str) -> str:
    """将 token 列表编码为 <|ID|> 格式字符串，附加说明文字。"""
    token_str = "".join(f"<|{t}|>" for t in tokens)
    suffix = (f" \n以上是该类{category}正常状态的特征映射。"
              "对比可见，当前产品存在异常。\n"
              "<|endoftext|><|im_end|>")
    return token_str + suffix


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
def stage1_extract_tokens(device: str = "cuda", target_categories: set = None) -> dict:
    """
    遍历 data/<category>/nomal/ 下所有图片，用 MoVQGAN 提取 384-token 序列。
    返回 {"category": {"img_id": [t1, t2, ...]}, ...}
    同时保存到 NORMAL_TOKEN_JSON。
    """
    sys.path.insert(0, str(MOVQGAN_DIR))
    from movqgan import get_movqgan_model  # noqa: E402

    print("=== 阶段一：加载 MoVQGAN 模型 ===")
    orig_cwd = os.getcwd()
    os.chdir(str(ROOT))
    model = get_movqgan_model("270M", pretrained=True, device=device)
    os.chdir(orig_cwd)

    all_tokens: dict = {}   # category -> {img_id: [tokens]}

    categories = [d.name for d in JUICE_ROOT.iterdir() if d.is_dir()]
    if target_categories is not None:
        categories = [c for c in categories if c in target_categories]
    for category in sorted(categories):
        normal_dir = JUICE_ROOT / category / "nomal"
        if not normal_dir.exists():
            print(f"  [跳过] {category} 无 nomal 目录")
            continue

        img_files = sorted(normal_dir.glob("*.png")) + sorted(normal_dir.glob("*.jpg"))
        cat_tokens: dict = {}

        print(f"  处理类别 [{category}]，共 {len(img_files)} 张正常图片")
        for img_path in tqdm(img_files, desc=f"  {category}"):
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

        # 每个类别单独保存 Token 文件
        token_json = MOVQGAN_DIR / f"gt_indices_{category}_normal.json"
        token_json.parent.mkdir(parents=True, exist_ok=True)
        flat = {img_id: toks for img_id, toks in cat_tokens.items()}
        with open(token_json, "w") as f:
            json.dump(flat, f, indent=4)
        print(f"  [{category}] Token 已保存至 {token_json}")
    print(f"[阶段一] 完成，共处理 {len(all_tokens)} 个类别\n")
    return all_tokens  # 嵌套结构供阶段三按类别抽取


# ──────────────────────────────────────────────────────────
# 阶段二：异常样本图像增强
# ──────────────────────────────────────────────────────────
def augment_image(img: Image.Image) -> Image.Image:
    """随机旋转 / 亮度 / 对比度 / 轻微裁剪。"""
    angle = random.uniform(-15, 15)
    img = TF.rotate(img, angle)

    brightness_factor = random.uniform(0.7, 1.3)
    img = TF.adjust_brightness(img, brightness_factor)

    contrast_factor = random.uniform(0.7, 1.3)
    img = TF.adjust_contrast(img, contrast_factor)

    w, h = img.size
    scale = random.uniform(0.85, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    left = random.randint(0, w - new_w)
    top  = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((w, h), Image.BICUBIC)

    return img


def _augment_worker(args):
    """多进程工作函数：读取、增强、保存单张图片。"""
    src_path, dst_path, seed = args
    random.seed(seed)
    img = Image.open(src_path).convert("RGB")
    aug = augment_image(img)
    aug.save(dst_path)
    return dst_path


def stage2_augment_anomalies(aug_count: int = 200, target_categories: set = None, num_workers: int = None) -> dict:
    """
    对每个类别的异常图片做增强，使每类总数达到 aug_count。
    增强后图片保存到 LLAMA_DATA/<category>_augmented/<category>/
    返回 {"category": [相对于 LLaMA-Factory/data/ 的路径, ...], ...}
    """
    print("=== 阶段二：异常样本图像增强 ===")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"  使用 {num_workers} 个进程并行处理")

    augmented_paths: dict = {}  # category -> [relative_path, ...]

    categories = [d.name for d in JUICE_ROOT.iterdir() if d.is_dir()]
    if target_categories is not None:
        categories = [c for c in categories if c in target_categories]
    for category in sorted(categories):
        anomal_dir = JUICE_ROOT / category / "anomal"
        if not anomal_dir.exists():
            print(f"  [跳过] {category} 无 anomal 目录")
            continue

        aug_dir = LLAMA_DATA / f"{category}_augmented" / category
        aug_dir.mkdir(parents=True, exist_ok=True)

        src_files = sorted(anomal_dir.glob("*.png")) + sorted(anomal_dir.glob("*.jpg"))
        if not src_files:
            print(f"  [跳过] {category}/anomal 无图片")
            continue

        cat_paths: list = []
        need = aug_count - len(src_files)
        total = len(src_files) + max(need, 0)
        print(f"  [{category}] 原始异常图 {len(src_files)} 张 → 目标 {total} 张")

        # 先把原始图复制过来
        for src in tqdm(src_files, desc=f"  {category} 复制原图"):
            dst = aug_dir / f"{category}_orig_{src.stem}.png"
            shutil.copy2(src, dst)
            cat_paths.append(str(dst.relative_to(LLAMA_DATA)))

        # 并行增强补足到 aug_count
        if need > 0:
            # 准备所有增强任务
            tasks = []
            for aug_idx in range(need):
                src = random.choice(src_files)
                aug_name = f"{category}_aug_{aug_idx:04d}_{src.stem}.png"
                dst = aug_dir / aug_name
                # 为每个任务生成不同的随机种子
                seed = 42 + aug_idx
                tasks.append((src, dst, seed))

            # 使用多进程池并行处理
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(_augment_worker, tasks),
                    total=len(tasks),
                    desc=f"  {category} 增强"
                ))

            # 收集结果路径
            for dst_path in results:
                cat_paths.append(str(Path(dst_path).relative_to(LLAMA_DATA)))

        augmented_paths[category] = cat_paths
        print(f"  [{category}] 完成，共 {len(cat_paths)} 张")

    print(f"[阶段二] 增强图片已保存至 {LLAMA_DATA}\n")
    return augmented_paths


# ──────────────────────────────────────────────────────────
# 阶段三：ShareGPT 格式数据集制作
# ──────────────────────────────────────────────────────────
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

        human_prompt = make_human_prompt(category)

        for img_rel_path in tqdm(img_paths, desc=f"  {category}"):
            # Category-Lock：从同类 Token 库随机抽取
            _, tokens = random.choice(cat_token_items)

            sample_id = f"{category}_{uuid.uuid4().hex[:13]}"
            gpt_value = tokens_to_gpt_value(tokens, category)

            record = {
                "id": sample_id,
                "images": [img_rel_path],
                "system": SYSTEM_PROMPT,
                "conversations": [
                    {"from": "human", "value": human_prompt},
                    {"from": "gpt",   "value": gpt_value},
                ],
            }
            dataset.append(record)

        print(f"  [{category}] 生成 {len(img_paths)} 条训练样本")

        # 每个类别单独保存数据集文件
        out_json = LLAMA_DATA / f"{category}_anomaly_sft.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        cat_records = [r for r in dataset if r["id"].startswith(category + "_")]
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(cat_records, f, ensure_ascii=False, indent=2)
        print(f"  [{category}] 数据集已保存至 {out_json}")

    print(f"[阶段三] 全部完成，共 {len(dataset)} 条\n")
    return dataset


# ──────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="异常检测数据集一键制作")
    parser.add_argument("--category", type=str, default=None,
                        help="指定处理的类别目录名（如 losiding），不指定则处理 data/ 下所有类别")
    parser.add_argument("--aug_count", type=int, default=200,
                        help="每类异常样本增强后目标总数（默认200）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch 设备（默认 cuda，无 GPU 可设为 cpu）")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="跳过阶段一（使用已有的 gt_indices_normal.json）")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="图像增强并行进程数（默认为 CPU 核心数-1）")
    args = parser.parse_args()

    if args.category is not None:
        cat_dir = JUICE_ROOT / args.category
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"指定的类别目录不存在: {cat_dir}")
        target_categories = {args.category}
    else:
        target_categories = None  # None 表示处理全部

    random.seed(42)

    # ── 阶段一 ──
    if args.skip_stage1:
        if not NORMAL_TOKEN_JSON.exists():
            raise FileNotFoundError(f"--skip_stage1 但未找到 {NORMAL_TOKEN_JSON}")
        print(f"[阶段一] 跳过，加载已有 Token 文件: {NORMAL_TOKEN_JSON}")
        with open(NORMAL_TOKEN_JSON) as f:
            loaded = json.load(f)
        # 兼容扁平格式 {"category_stem": [tokens]} → 嵌套
        first_val = next(iter(loaded.values()))
        if isinstance(first_val, list):
            # 用所有已知类别目录名作为分隔依据，避免 split("_")[0] 误切
            known_categories = {d.name for d in JUICE_ROOT.iterdir() if d.is_dir()}
            all_tokens: dict = {}
            for img_id, tokens in loaded.items():
                # 找到最长匹配的类别前缀
                cat = next(
                    (c for c in sorted(known_categories, key=len, reverse=True)
                     if img_id.startswith(c + "_")),
                    img_id.split("_")[0]  # fallback
                )
                all_tokens.setdefault(cat, {})[img_id] = tokens
        else:
            all_tokens = loaded
    else:
        all_tokens = stage1_extract_tokens(device=args.device, target_categories=target_categories)

    # --skip_stage1 时也按 target_categories 过滤 all_tokens
    if args.skip_stage1 and target_categories is not None:
        all_tokens = {k: v for k, v in all_tokens.items() if k in target_categories}

    # ── 阶段二 ──
    augmented_paths = stage2_augment_anomalies(aug_count=args.aug_count, target_categories=target_categories, num_workers=args.num_workers)

    # ── 阶段三 ──
    stage3_build_sharegpt(all_tokens, augmented_paths)

    print("=== 全流程完成 ===")
    print(f"  Token 文件目录  : {MOVQGAN_DIR}")
    print(f"  增强图片目录    : {LLAMA_DATA}")
    print(f"  数据集目录      : {LLAMA_DATA}")


if __name__ == "__main__":
    main()
