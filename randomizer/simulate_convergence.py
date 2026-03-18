#!/usr/bin/env python3
"""
模拟模型推理收敛过程：从完全随机逐步接近正确输出
生成10次推理的数据，每次推理逐步减少随机范围
"""

import json
import re
import random
import os
from pathlib import Path


def extract_tokens(text):
    """提取文本中所有的<|xxxx|>格式的token"""
    pattern = r'<\|(\d+)\|>'
    matches = re.finditer(pattern, text)
    tokens = [(match.group(0), match.start(), match.end(), int(match.group(1)))
              for match in matches]
    return tokens


def randomize_positions(text, positions, min_val=0, max_val=16383):
    """随机化指定位置的token"""
    tokens = extract_tokens(text)

    # 构建新文本
    result = text
    offset = 0

    for i in sorted(positions):
        if i >= len(tokens):
            continue

        token_str, start_idx, end_idx, old_val = tokens[i]
        new_val = random.randint(min_val, max_val)
        new_token = f'<|{new_val}|>'

        # 计算实际位置（考虑之前的替换导致的偏移）
        actual_start = start_idx + offset
        actual_end = end_idx + offset

        result = result[:actual_start] + new_token + result[actual_end:]
        offset += len(new_token) - len(token_str)

    return result


def design_convergence_strategy():
    """
    设计10次推理的收敛策略
    从完全随机逐步收敛到正确输出

    返回：每次推理需要随机化的位置集合
    """
    total_positions = 384
    strategies = []

    # 第1次：完全随机（100%）
    strategies.append({
        'iteration': 1,
        'description': '完全随机（100%随机）',
        'positions': set(range(0, 384))
    })

    # 第2次：保留开头和结尾各10%，随机中间80%
    strategies.append({
        'iteration': 2,
        'description': '保留开头和结尾各10%，随机中间80%',
        'positions': set(range(38, 346))
    })

    # 第3次：保留开头和结尾各20%，随机中间60%
    strategies.append({
        'iteration': 3,
        'description': '保留开头和结尾各20%，随机中间60%',
        'positions': set(range(77, 307))
    })

    # 第4次：保留开头和结尾各30%，随机中间40%
    strategies.append({
        'iteration': 4,
        'description': '保留开头和结尾各30%，随机中间40%',
        'positions': set(range(115, 269))
    })

    # 第5次：保留开头和结尾各35%，随机中间30%
    strategies.append({
        'iteration': 5,
        'description': '保留开头和结尾各35%，随机中间30%',
        'positions': set(range(134, 250))
    })

    # 第6次：随机分散的4个区域（约25%）
    positions_6 = set()
    positions_6.update(range(20, 50))      # 30个
    positions_6.update(range(100, 130))    # 30个
    positions_6.update(range(200, 220))    # 20个
    positions_6.update(range(300, 320))    # 20个
    strategies.append({
        'iteration': 6,
        'description': '随机分散的4个区域（约26%）',
        'positions': positions_6
    })

    # 第7次：随机更少的3个区域（约15%）
    positions_7 = set()
    positions_7.update(range(50, 75))      # 25个
    positions_7.update(range(150, 170))    # 20个
    positions_7.update(range(280, 295))    # 15个
    strategies.append({
        'iteration': 7,
        'description': '随机3个较小区域（约16%）',
        'positions': positions_7
    })

    # 第8次：只随机2个小区域（约8%）
    positions_8 = set()
    positions_8.update(range(80, 95))      # 15个
    positions_8.update(range(250, 265))    # 15个
    strategies.append({
        'iteration': 8,
        'description': '随机2个小区域（约8%）',
        'positions': positions_8
    })

    # 第9次：只随机很少的位置（约3%）
    positions_9 = set()
    positions_9.update(range(120, 130))    # 10个
    strategies.append({
        'iteration': 9,
        'description': '随机1个很小区域（约3%）',
        'positions': positions_9
    })

    # 第10次：几乎完全正确（只随机1%）
    positions_10 = set()
    positions_10.update(range(190, 195))   # 5个
    strategies.append({
        'iteration': 10,
        'description': '几乎完全正确（只随机1%）',
        'positions': positions_10
    })

    return strategies


def simulate_convergence(input_file, output_dir, seed=None):
    """
    模拟10次推理的收敛过程

    Args:
        input_file: 输入文件（正确的输出）
        output_dir: 输出目录
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取正确的输出
    with open(input_file, 'r', encoding='utf-8') as f:
        correct_data = json.load(f)

    # 获取收敛策略
    strategies = design_convergence_strategy()

    print("=" * 60)
    print("模拟模型推理收敛过程")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"总共生成: {len(strategies)} 次推理结果")
    print("=" * 60)
    print()

    # 生成每次推理的数据
    for strategy in strategies:
        iteration = strategy['iteration']
        description = strategy['description']
        positions = strategy['positions']

        print(f"第{iteration}次推理: {description}")
        print(f"  随机化位置数: {len(positions)}")
        print(f"  随机化比例: {len(positions)/384*100:.1f}%")

        # 创建这次推理的数据
        iteration_data = {}
        for key, correct_value in correct_data.items():
            # 从正确输出开始，随机化指定位置
            iteration_data[key] = randomize_positions(correct_value, positions)

        # 保存结果
        output_file = os.path.join(output_dir, f'iteration_{iteration:02d}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(iteration_data, f, ensure_ascii=False, indent=4)

        print(f"  已保存: {output_file}")
        print()

    print("=" * 60)
    print("收敛过程模拟完成！")
    print("=" * 60)

    # 生成收敛策略说明文件
    strategy_file = os.path.join(output_dir, 'convergence_strategy.txt')
    with open(strategy_file, 'w', encoding='utf-8') as f:
        f.write("模型推理收敛策略说明\n")
        f.write("=" * 60 + "\n\n")
        for strategy in strategies:
            f.write(f"第{strategy['iteration']}次推理:\n")
            f.write(f"  描述: {strategy['description']}\n")
            f.write(f"  随机化位置数: {len(strategy['positions'])}\n")
            f.write(f"  随机化比例: {len(strategy['positions'])/384*100:.1f}%\n")
            f.write(f"  保留正确值比例: {(384-len(strategy['positions']))/384*100:.1f}%\n")

            # 显示随机化的范围
            positions_list = sorted(strategy['positions'])
            if positions_list:
                ranges = []
                start = positions_list[0]
                end = positions_list[0]

                for pos in positions_list[1:]:
                    if pos == end + 1:
                        end = pos
                    else:
                        ranges.append(f"{start}-{end}" if start != end else f"{start}")
                        start = pos
                        end = pos
                ranges.append(f"{start}-{end}" if start != end else f"{start}")

                f.write(f"  随机化范围: {', '.join(ranges)}\n")
            f.write("\n")

    print(f"收敛策略说明已保存: {strategy_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='模拟模型推理收敛过程')
    parser.add_argument('input', help='输入JSON文件（正确的输出）')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')

    args = parser.parse_args()

    simulate_convergence(args.input, args.output_dir, args.seed)
