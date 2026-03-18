#!/bin/bash
# 批量处理脚本：对指定目录下的所有JSON文件进行随机化处理

# 使用说明
usage() {
    echo "用法: $0 <输入目录> <输出目录> <模式> [其他参数]"
    echo ""
    echo "模式:"
    echo "  all         - 全部随机"
    echo "  continuous  - 连续随机（需要 --start 和 --length）"
    echo "  range       - 范围随机（需要 --start 和 --end）"
    echo ""
    echo "示例:"
    echo "  $0 data_test/ output/ all"
    echo "  $0 data_test/ output/ continuous --start 10 --length 20"
    echo "  $0 data_test/ output/ range --start 50 --end 100"
    exit 1
}

# 检查参数
if [ $# -lt 3 ]; then
    usage
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
MODE=$3
shift 3  # 移除前三个参数，剩下的作为额外参数

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 '$INPUT_DIR' 不存在"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 统计变量
total_files=0
success_files=0
failed_files=0

echo "=========================================="
echo "批量处理JSON文件"
echo "=========================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "处理模式: $MODE"
echo "额外参数: $@"
echo "=========================================="
echo ""

# 遍历输入目录中的所有JSON文件
for input_file in "$INPUT_DIR"/*.json; do
    # 检查文件是否存在（处理没有匹配文件的情况）
    if [ ! -f "$input_file" ]; then
        echo "警告: 在 '$INPUT_DIR' 中没有找到JSON文件"
        break
    fi

    # 获取文件名
    filename=$(basename "$input_file")
    output_file="$OUTPUT_DIR/$filename"

    echo "处理: $filename"

    # 执行随机化
    if python3 randomize_tokens.py "$input_file" "$output_file" --mode "$MODE" "$@"; then
        ((success_files++))
        echo "  ✓ 成功"
    else
        ((failed_files++))
        echo "  ✗ 失败"
    fi

    ((total_files++))
    echo ""
done

# 输出统计信息
echo "=========================================="
echo "处理完成"
echo "=========================================="
echo "总文件数: $total_files"
echo "成功: $success_files"
echo "失败: $failed_files"
echo "=========================================="
