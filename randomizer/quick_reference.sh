#!/bin/bash
# 快速参考：常用命令示例

echo "=========================================="
echo "Token随机化脚本 - 快速参考"
echo "=========================================="
echo ""

echo "【基本用法】"
echo ""
echo "1. 全部随机（所有384个位置）："
echo "   python3 randomize_tokens.py input.json output.json --mode all"
echo ""

echo "2. 连续随机（从位置10开始，连续20个）："
echo "   python3 randomize_tokens.py input.json output.json --mode continuous --start 10 --length 20"
echo ""

echo "3. 范围随机（第50到第100个位置）："
echo "   python3 randomize_tokens.py input.json output.json --mode range --start 50 --end 100"
echo ""

echo "4. 多范围随机（多个不连续的范围）⭐新功能："
echo "   python3 randomize_tokens.py input.json output.json --mode multi-range --ranges \"2-60,200-270,300-320,350-383\""
echo ""

echo "=========================================="
echo "【批量处理】"
echo "=========================================="
echo ""

echo "处理整个目录的所有JSON文件："
echo "   ./batch_randomize.sh data_test/ output/ all"
echo ""

echo "批量处理使用多范围模式："
echo "   ./batch_randomize.sh data_test/ output/ multi-range --ranges \"2-60,200-270,300-320,350-383\""
echo ""

echo "=========================================="
echo "【测试和验证】"
echo "=========================================="
echo ""

echo "运行测试（生成示例输出）："
echo "   ./test_randomize.sh"
echo ""

echo "验证输出结果："
echo "   python3 verify_output.py"
echo ""

echo "验证多范围结果："
echo "   python3 verify_multi_range.py"
echo ""

echo "=========================================="
echo "【高级选项】"
echo "=========================================="
echo ""

echo "使用固定随机种子（可重复结果）："
echo "   python3 randomize_tokens.py input.json output.json --mode all --seed 42"
echo ""

echo "自定义随机数范围："
echo "   python3 randomize_tokens.py input.json output.json --mode all --min 1000 --max 15000"
echo ""

echo "=========================================="
echo "【多范围模式应用场景】"
echo "=========================================="
echo ""
echo "场景A：只随机化开头和结尾"
echo "   --ranges \"0-50,330-383\""
echo ""
echo "场景B：随机化四个均匀分布的区域"
echo "   --ranges \"0-95,96-191,192-287,288-383\""
echo ""
echo "场景C：保留关键位置100-110，随机化其他"
echo "   --ranges \"0-99,111-383\""
echo ""
echo "场景D：自定义任意组合"
echo "   --ranges \"2-60,200-270,300-320,350-383\""
echo ""

echo "=========================================="
echo "【位置说明】"
echo "=========================================="
echo ""
echo "- 位置索引从0开始，范围是 [0, 383]"
echo "- 总共有384个token位置"
echo "- range和multi-range模式中，end位置是包含的"
echo "- 随机数范围默认是 [0, 16383]"
echo "- 多范围可以重叠，会自动去重"
echo ""

echo "=========================================="
echo "详细文档请查看: TOKEN_RANDOMIZE_README.md"
echo "=========================================="
