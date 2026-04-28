#!/usr/bin/env python3
"""
验证多范围随机化结果
"""

import json
import re


def verify_multi_range(original_file, processed_file, ranges_str):
    """验证多范围随机化结果"""
    # 解析范围
    ranges = []
    for part in ranges_str.split(','):
        start, end = map(int, part.strip().split('-'))
        ranges.append((start, end))

    # 收集应该被随机化的位置
    expected_positions = set()
    for start, end in ranges:
        for i in range(start, end + 1):
            expected_positions.add(i)

    print(f"预期随机化的位置数: {len(expected_positions)}")
    print(f"范围: {ranges_str}")
    print()

    # 读取文件
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    # 验证每条数据
    for key in original_data.keys():
        original_tokens = re.findall(r'<\|(\d+)\|>', original_data[key])
        processed_tokens = re.findall(r'<\|(\d+)\|>', processed_data[key])

        print(f"数据ID: {key[:32]}...")
        print(f"  原始token数: {len(original_tokens)}")
        print(f"  处理后token数: {len(processed_tokens)}")

        # 统计变化的位置
        changed_positions = set()
        unchanged_positions = set()

        for i, (orig, proc) in enumerate(zip(original_tokens, processed_tokens)):
            if orig != proc:
                changed_positions.add(i)
            else:
                unchanged_positions.add(i)

        print(f"  实际变化的位置数: {len(changed_positions)}")
        print(f"  未变化的位置数: {len(unchanged_positions)}")

        # 检查是否符合预期
        if changed_positions == expected_positions:
            print(f"  ✓ 验证通过：变化的位置与预期完全一致")
        else:
            print(f"  ✗ 验证失败：")
            unexpected_changes = changed_positions - expected_positions
            missing_changes = expected_positions - changed_positions

            if unexpected_changes:
                print(f"    意外变化的位置: {sorted(list(unexpected_changes))[:10]}...")
            if missing_changes:
                print(f"    应该变化但未变化的位置: {sorted(list(missing_changes))[:10]}...")

        # 显示一些示例
        print(f"  示例变化:")
        count = 0
        for pos in sorted(expected_positions):
            if pos < len(original_tokens) and count < 5:
                print(f"    位置{pos}: {original_tokens[pos]} -> {processed_tokens[pos]}")
                count += 1

        print()


if __name__ == '__main__':
    verify_multi_range(
        'data_test/eval_traj_100.json',
        'test_multi_range.json',
        '2-60,200-270,300-320,350-383'
    )
