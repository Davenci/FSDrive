#!/usr/bin/env python3
"""
数据编辑脚本：随机替换JSON文件中的特殊token <|xxxx|>
支持四种替换模式：
1. 全部随机：所有384个位置的数字都随机
2. 连续随机：从指定起始位置开始，连续n个位置随机
3. 范围随机：指定第n个到第k个位置随机
4. 多范围随机：指定多个不连续的范围随机（如：2-60,200-270,300-320,350-384）
"""

import json
import re
import random
import argparse
from pathlib import Path


def extract_tokens(text):
    """提取文本中所有的<|xxxx|>格式的token"""
    pattern = r'<\|(\d+)\|>'
    matches = re.finditer(pattern, text)
    tokens = [(match.group(0), match.start(), match.end(), int(match.group(1)))
              for match in matches]
    return tokens


def randomize_all(text, min_val=0, max_val=16383):
    """模式1：所有384个位置的数字都随机"""
    def replace_func(match):
        random_num = random.randint(min_val, max_val)
        return f'<|{random_num}|>'

    pattern = r'<\|\d+\|>'
    new_text = re.sub(pattern, replace_func, text)
    return new_text


def randomize_continuous(text, start_pos, length, min_val=0, max_val=16383):
    """模式2：从start_pos开始，连续length个位置随机"""
    tokens = extract_tokens(text)

    if start_pos < 0 or start_pos >= len(tokens):
        raise ValueError(f"起始位置 {start_pos} 超出范围 [0, {len(tokens)-1}]")

    end_pos = min(start_pos + length, len(tokens))

    # 构建新文本
    result = text
    offset = 0

    for i in range(start_pos, end_pos):
        token_str, start_idx, end_idx, old_val = tokens[i]
        new_val = random.randint(min_val, max_val)
        new_token = f'<|{new_val}|>'

        # 计算实际位置（考虑之前的替换导致的偏移）
        actual_start = start_idx + offset
        actual_end = end_idx + offset

        result = result[:actual_start] + new_token + result[actual_end:]
        offset += len(new_token) - len(token_str)

    return result


def randomize_range(text, start_pos, end_pos, min_val=0, max_val=16383):
    """模式3：指定第start_pos个到第end_pos个位置随机（包含end_pos）"""
    tokens = extract_tokens(text)

    if start_pos < 0 or start_pos >= len(tokens):
        raise ValueError(f"起始位置 {start_pos} 超出范围 [0, {len(tokens)-1}]")
    if end_pos < 0 or end_pos >= len(tokens):
        raise ValueError(f"结束位置 {end_pos} 超出范围 [0, {len(tokens)-1}]")
    if start_pos > end_pos:
        raise ValueError(f"起始位置 {start_pos} 不能大于结束位置 {end_pos}")

    # 构建新文本
    result = text
    offset = 0

    for i in range(start_pos, end_pos + 1):
        token_str, start_idx, end_idx, old_val = tokens[i]
        new_val = random.randint(min_val, max_val)
        new_token = f'<|{new_val}|>'

        # 计算实际位置（考虑之前的替换导致的偏移）
        actual_start = start_idx + offset
        actual_end = end_idx + offset

        result = result[:actual_start] + new_token + result[actual_end:]
        offset += len(new_token) - len(token_str)

    return result


def parse_ranges(ranges_str):
    """解析范围字符串，如 "2-60,200-270,300-320,350-384" """
    ranges = []
    parts = ranges_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' not in part:
            raise ValueError(f"范围格式错误: {part}，应该是 'start-end' 格式")

        start_str, end_str = part.split('-', 1)
        try:
            start = int(start_str.strip())
            end = int(end_str.strip())
        except ValueError:
            raise ValueError(f"范围格式错误: {part}，起始和结束位置必须是整数")

        if start > end:
            raise ValueError(f"范围错误: {part}，起始位置不能大于结束位置")

        ranges.append((start, end))

    return ranges


def randomize_multi_range(text, ranges_str, min_val=0, max_val=16383):
    """模式4：指定多个不连续的范围随机

    Args:
        text: 输入文本
        ranges_str: 范围字符串，格式如 "2-60,200-270,300-320,350-384"
        min_val: 随机数最小值
        max_val: 随机数最大值

    Returns:
        处理后的文本
    """
    tokens = extract_tokens(text)
    ranges = parse_ranges(ranges_str)

    # 验证所有范围
    for start, end in ranges:
        if start < 0 or start >= len(tokens):
            raise ValueError(f"起始位置 {start} 超出范围 [0, {len(tokens)-1}]")
        if end < 0 or end >= len(tokens):
            raise ValueError(f"结束位置 {end} 超出范围 [0, {len(tokens)-1}]")

    # 收集所有需要随机化的位置（去重并排序）
    positions_to_randomize = set()
    for start, end in ranges:
        for i in range(start, end + 1):
            positions_to_randomize.add(i)

    positions_to_randomize = sorted(positions_to_randomize)

    # 构建新文本
    result = text
    offset = 0

    for i in positions_to_randomize:
        token_str, start_idx, end_idx, old_val = tokens[i]
        new_val = random.randint(min_val, max_val)
        new_token = f'<|{new_val}|>'

        # 计算实际位置（考虑之前的替换导致的偏移）
        actual_start = start_idx + offset
        actual_end = end_idx + offset

        result = result[:actual_start] + new_token + result[actual_end:]
        offset += len(new_token) - len(token_str)

    print(f"多范围随机化: 共处理 {len(positions_to_randomize)} 个位置")
    print(f"范围: {ranges_str}")

    return result


def process_json_file(input_file, output_file, mode, **kwargs):
    """处理JSON文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理每条数据
    for key, value in data.items():
        if mode == 'all':
            data[key] = randomize_all(value, kwargs.get('min_val', 0), kwargs.get('max_val', 16383))
        elif mode == 'continuous':
            start_pos = kwargs.get('start_pos', 0)
            length = kwargs.get('length', 10)
            data[key] = randomize_continuous(value, start_pos, length,
                                            kwargs.get('min_val', 0), kwargs.get('max_val', 16383))
        elif mode == 'range':
            start_pos = kwargs.get('start_pos', 0)
            end_pos = kwargs.get('end_pos', 10)
            data[key] = randomize_range(value, start_pos, end_pos,
                                       kwargs.get('min_val', 0), kwargs.get('max_val', 16383))
        elif mode == 'multi-range':
            ranges_str = kwargs.get('ranges_str')
            if not ranges_str:
                raise ValueError("multi-range模式需要提供 --ranges 参数")
            data[key] = randomize_multi_range(value, ranges_str,
                                             kwargs.get('min_val', 0), kwargs.get('max_val', 16383))
        else:
            raise ValueError(f"未知的模式: {mode}")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"处理完成！结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='随机替换JSON文件中的特殊token <|xxxx|>')
    parser.add_argument('input', help='输入JSON文件路径')
    parser.add_argument('output', help='输出JSON文件路径')
    parser.add_argument('--mode', choices=['all', 'continuous', 'range', 'multi-range'], required=True,
                       help='替换模式: all(全部随机), continuous(连续随机), range(范围随机), multi-range(多范围随机)')
    parser.add_argument('--start', type=int, default=0,
                       help='起始位置（用于continuous和range模式）')
    parser.add_argument('--length', type=int, default=10,
                       help='连续长度（用于continuous模式）')
    parser.add_argument('--end', type=int, default=10,
                       help='结束位置（用于range模式）')
    parser.add_argument('--ranges', type=str, default=None,
                       help='多个范围（用于multi-range模式），格式如: "2-60,200-270,300-320,350-383"')
    parser.add_argument('--min', type=int, default=0,
                       help='随机数最小值（默认0）')
    parser.add_argument('--max', type=int, default=16383,
                       help='随机数最大值（默认16383）')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（可选，用于可重复的随机结果）')

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)

    # 处理文件
    process_json_file(
        args.input,
        args.output,
        args.mode,
        start_pos=args.start,
        length=args.length,
        end_pos=args.end,
        ranges_str=args.ranges,
        min_val=args.min,
        max_val=args.max
    )


if __name__ == '__main__':
    main()
