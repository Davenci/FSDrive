#!/usr/bin/env python3
"""
验证收敛过程生成的数据
"""

import json
import re
import os


def verify_convergence(correct_file, convergence_dir):
    """验证收敛过程的数据"""

    # 读取正确的输出
    with open(correct_file, 'r', encoding='utf-8') as f:
        correct_data = json.load(f)

    print("=" * 70)
    print("验证收敛过程数据")
    print("=" * 70)
    print(f"正确输出文件: {correct_file}")
    print(f"收敛数据目录: {convergence_dir}")
    print("=" * 70)
    print()

    # 遍历10次推理的结果
    for iteration in range(1, 11):
        iteration_file = os.path.join(convergence_dir, f'iteration_{iteration:02d}.json')

        if not os.path.exists(iteration_file):
            print(f"第{iteration}次推理: 文件不存在")
            continue

        with open(iteration_file, 'r', encoding='utf-8') as f:
            iteration_data = json.load(f)

        print(f"第{iteration}次推理:")

        # 验证每条数据
        for key in correct_data.keys():
            correct_tokens = re.findall(r'<\|(\d+)\|>', correct_data[key])
            iteration_tokens = re.findall(r'<\|(\d+)\|>', iteration_data[key])

            # 统计相同和不同的位置
            same_count = sum(1 for c, i in zip(correct_tokens, iteration_tokens) if c == i)
            diff_count = len(correct_tokens) - same_count

            accuracy = same_count / len(correct_tokens) * 100

            print(f"  数据 {key[:32]}...")
            print(f"    Token总数: {len(correct_tokens)}")
            print(f"    与正确输出相同: {same_count} ({accuracy:.1f}%)")
            print(f"    与正确输出不同: {diff_count} ({100-accuracy:.1f}%)")

        print()

    print("=" * 70)
    print("验证完成")
    print("=" * 70)


def visualize_convergence(correct_file, convergence_dir):
    """可视化收敛过程"""

    # 读取正确的输出
    with open(correct_file, 'r', encoding='utf-8') as f:
        correct_data = json.load(f)

    # 获取第一条数据的key
    first_key = list(correct_data.keys())[0]
    correct_tokens = re.findall(r'<\|(\d+)\|>', correct_data[first_key])

    print("\n" + "=" * 70)
    print("收敛过程可视化（准确率变化）")
    print("=" * 70)
    print()

    accuracies = []

    for iteration in range(1, 11):
        iteration_file = os.path.join(convergence_dir, f'iteration_{iteration:02d}.json')

        if not os.path.exists(iteration_file):
            continue

        with open(iteration_file, 'r', encoding='utf-8') as f:
            iteration_data = json.load(f)

        iteration_tokens = re.findall(r'<\|(\d+)\|>', iteration_data[first_key])
        same_count = sum(1 for c, i in zip(correct_tokens, iteration_tokens) if c == i)
        accuracy = same_count / len(correct_tokens) * 100

        accuracies.append(accuracy)

        # 绘制简单的进度条
        bar_length = 50
        filled_length = int(bar_length * accuracy / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(f"第{iteration:2d}次推理: [{bar}] {accuracy:5.1f}%")

    print()
    print("=" * 70)
    print("收敛趋势:")
    print(f"  起始准确率: {accuracies[0]:.1f}%")
    print(f"  最终准确率: {accuracies[-1]:.1f}%")
    print(f"  准确率提升: {accuracies[-1] - accuracies[0]:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='验证收敛过程数据')
    parser.add_argument('correct_file', help='正确输出文件')
    parser.add_argument('convergence_dir', help='收敛数据目录')

    args = parser.parse_args()

    verify_convergence(args.correct_file, args.convergence_dir)
    visualize_convergence(args.correct_file, args.convergence_dir)
