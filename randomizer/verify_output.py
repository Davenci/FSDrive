#!/usr/bin/env python3
"""
验证脚本：检查处理后的文件是否正确
"""

import json
import re


def count_tokens(text):
    """统计文本中<|xxxx|>格式的token数量"""
    pattern = r'<\|\d+\|>'
    matches = re.findall(pattern, text)
    return len(matches)


def verify_file(file_path):
    """验证文件中每条数据的token数量"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n文件: {file_path}")
    print("-" * 60)

    for key, value in data.items():
        token_count = count_tokens(value)
        print(f"  {key[:32]}... : {token_count} tokens")

        # 提取前5个和后5个token作为示例
        pattern = r'<\|(\d+)\|>'
        tokens = re.findall(pattern, value)
        if len(tokens) >= 10:
            print(f"    前5个: {tokens[:5]}")
            print(f"    后5个: {tokens[-5:]}")
        print()


def compare_files(original_file, processed_file):
    """比较原始文件和处理后的文件"""
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    print(f"\n比较: {original_file} vs {processed_file}")
    print("=" * 60)

    for key in original_data.keys():
        original_tokens = re.findall(r'<\|(\d+)\|>', original_data[key])
        processed_tokens = re.findall(r'<\|(\d+)\|>', processed_data[key])

        # 统计变化的token数量
        changed_count = sum(1 for o, p in zip(original_tokens, processed_tokens) if o != p)

        print(f"\n  {key[:32]}...")
        print(f"    原始token数: {len(original_tokens)}")
        print(f"    处理后token数: {len(processed_tokens)}")
        print(f"    变化的token数: {changed_count}")

        # 显示前3个变化的位置
        changes = [(i, o, p) for i, (o, p) in enumerate(zip(original_tokens, processed_tokens)) if o != p]
        if changes:
            print(f"    前3个变化:")
            for i, old, new in changes[:3]:
                print(f"      位置{i}: {old} -> {new}")


if __name__ == '__main__':
    print("=" * 60)
    print("验证处理后的文件")
    print("=" * 60)

    # 验证所有输出文件
    verify_file('data_test_output/eval_traj_100_all.json')
    verify_file('data_test_output/eval_traj_100_continuous.json')
    verify_file('data_test_output/eval_traj_100_range.json')
    verify_file('data_test_output/eval_traj_1800_all.json')

    # 比较原始文件和处理后的文件
    print("\n" + "=" * 60)
    print("比较原始文件和处理后的文件")
    print("=" * 60)

    compare_files('data_test/eval_traj_100.json', 'data_test_output/eval_traj_100_all.json')
    compare_files('data_test/eval_traj_100.json', 'data_test_output/eval_traj_100_continuous.json')
    compare_files('data_test/eval_traj_100.json', 'data_test_output/eval_traj_100_range.json')
