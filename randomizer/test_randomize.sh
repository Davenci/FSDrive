#!/bin/bash
# 测试脚本：演示四种随机替换模式

echo "=== 测试数据编辑脚本 ==="
echo ""

# 创建输出目录
mkdir -p data_test_output

echo "1. 测试模式1：全部随机（所有384个位置）"
python3 randomize_tokens.py \
    data_test/eval_traj_100.json \
    data_test_output/eval_traj_100_all.json \
    --mode all \
    --seed 42
echo ""

echo "2. 测试模式2：连续随机（从位置10开始，连续20个位置）"
python3 randomize_tokens.py \
    data_test/eval_traj_100.json \
    data_test_output/eval_traj_100_continuous.json \
    --mode continuous \
    --start 10 \
    --length 20 \
    --seed 42
echo ""

echo "3. 测试模式3：范围随机（第50到第100个位置）"
python3 randomize_tokens.py \
    data_test/eval_traj_100.json \
    data_test_output/eval_traj_100_range.json \
    --mode range \
    --start 50 \
    --end 100 \
    --seed 42
echo ""

echo "4. 测试模式4：多范围随机（2-60,200-270,300-320,350-383）"
python3 randomize_tokens.py \
    data_test/eval_traj_100.json \
    data_test_output/eval_traj_100_multi_range.json \
    --mode multi-range \
    --ranges "2-60,200-270,300-320,350-383" \
    --seed 42
echo ""

echo "5. 测试第二个文件：全部随机"
python3 randomize_tokens.py \
    data_test/eval_traj_1800.json \
    data_test_output/eval_traj_1800_all.json \
    --mode all \
    --seed 42
echo ""

echo "=== 测试完成 ==="
echo "输出文件保存在 data_test_output/ 目录中"
