"""
使用说明:
    python eval_and_vis.py \
        --pred_trajs_path ./LLaMA-Factory/results_losiding_50.jsonl \
        --token_traj_path ./LLaMA-Factory/data/losiding_30_val.json \
        --output_dir ./losiding_vis_50

    可选参数:
        --save_eval_json <path>   保存中间生成的 eval_traj.json（默认不保存）

    功能说明:
        整合 tools/match.py 和 MoVQGAN/vis.py 两个脚本的处理逻辑：
        Step 1 (match): 从预测结果 JSONL 和 token 轨迹 JSON 构建评估轨迹字典
        Step 2 (vis):   使用 MoVQGAN 模型将 token 序列解码为图像，保存到输出目录
"""

import os
import re
import sys
import json
import argparse

# ── Step 1: tools/match.py 逻辑 ──────────────────────────────────────────────

def load_pred_trajs_from_json(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_eval_traj(pred_trajs_path, token_traj_path):
    pred_trajs = load_pred_trajs_from_json(pred_trajs_path)
    token_traj = json.load(open(token_traj_path, 'r'))

    eval_traj = {}
    token_traj = token_traj[:100]  # 与 match.py 保持一致：只评估前100个轨迹
    for i, traj in enumerate(token_traj):
        eval_traj[traj['id']] = pred_trajs[i]['predict']

    return eval_traj


# ── Step 2: MoVQGAN/vis.py 逻辑 ──────────────────────────────────────────────

def visualize_traj(eval_traj, output_dir):
    import torch
    from PIL import Image

    # 将 MoVQGAN 目录加入路径，以便找到 movqgan 包
    movqgan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MoVQGAN')
    if movqgan_dir not in sys.path:
        sys.path.insert(0, movqgan_dir)

    from movqgan import get_movqgan_model

    def show_images(batch, file_path):
        scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        image = Image.fromarray(reshaped.numpy())
        image.save(file_path)

    model = get_movqgan_model('270M', pretrained=True, device='cuda')
    os.makedirs(output_dir, exist_ok=True)

    for key, value in eval_traj.items():
        try:
            token = key
            idx = value
            numbers = re.findall(r'<\|(\d+)\|>', idx)
            idx = [int(num) for num in numbers]
            idx = torch.tensor(idx).to(model.device)
            idx = torch.clamp(idx, min=0, max=16383)

            current_length = idx.size(0)
            required_length = 384

            if current_length < required_length:
                pad_length = required_length - current_length
                padding = torch.randint(0, 16384, (pad_length,), device=idx.device, dtype=idx.dtype)
                idx = torch.cat([idx, padding], dim=0)

            with torch.no_grad():
                out = model.decode_code(idx[:required_length].unsqueeze(0))
            save_path = os.path.join(output_dir, f"{token}.png")
            show_images(out, save_path)
        except Exception:
            continue


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline: match predictions -> visualize trajectories')
    parser.add_argument('--pred_trajs_path', type=str, required=True,
                        help='预测结果 JSONL 文件路径 (e.g., results_losiding_50.jsonl)')
    parser.add_argument('--token_traj_path', type=str, required=True,
                        help='token 轨迹 JSON 文件路径 (e.g., losiding_30_val.json)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='图像输出目录')
    parser.add_argument('--save_eval_json', type=str, default=None,
                        help='（可选）保存中间 eval_traj.json 的路径')
    args = parser.parse_args()

    print("[1/2] 生成评估轨迹 (match)...")
    eval_traj = build_eval_traj(args.pred_trajs_path, args.token_traj_path)
    print(f"      共 {len(eval_traj)} 条轨迹")

    if args.save_eval_json:
        with open(args.save_eval_json, 'w') as f:
            json.dump(eval_traj, f, indent=4)
        print(f"      eval_traj 已保存至 {args.save_eval_json}")

    print("[2/2] 可视化轨迹 (vis)...")
    visualize_traj(eval_traj, args.output_dir)
    print(f"      图像已保存至 {args.output_dir}")
