# Token随机化脚本使用说明

## 功能概述

这个脚本用于随机替换JSON文件中的特殊token `<|xxxx|>`，支持四种不同的随机化模式。

## 文件说明

- `randomize_tokens.py` - 主脚本
- `test_randomize.sh` - 测试脚本（演示四种模式）
- `verify_output.py` - 验证脚本（检查输出结果）
- `verify_multi_range.py` - 多范围验证脚本
- `batch_randomize.sh` - 批量处理脚本

## 四种随机化模式

### 模式1：全部随机 (all)
所有384个位置的数字都随机替换。

**使用示例：**
```bash
python3 randomize_tokens.py \
    data_test/eval_traj_100.json \
    output/result.json \
    --mode all
```

### 模式2：连续随机 (continuous)
从指定起始位置开始，连续n个位置随机替换，其他位置保持不变。

**使用示例：**
```bash
# 从位置10开始，连续20个位置随机
python3 randomize_tokens.py \
    data_test/eval_traj_100.json \
    output/result.json \
    --mode continuous \
    --start 10 \
    --length 20
```

### 模式3：范围随机 (range)
指定第n个到第k个位置随机替换（包含第k个位置）。

**使用示例：**
```bash
# 第50到第100个位置随机（共51个位置）
python3 randomize_tokens.py \
    data_test/eval_traj_100.json \
    output/result.json \
    --mode range \
    --start 50 \
    --end 100
```

### 模式4：多范围随机 (multi-range) ⭐新功能
一次指定多个不连续的范围进行随机替换。可以灵活地选择任意多个范围组合。

**使用示例：**
```bash
# 同时随机化多个不连续的范围：2-60, 200-270, 300-320, 350-383
python3 randomize_tokens.py \
    data_test/eval_traj_100.json \
    output/result.json \
    --mode multi-range \
    --ranges "2-60,200-270,300-320,350-383"
```

**范围格式说明：**
- 使用逗号分隔多个范围
- 每个范围格式为 "起始位置-结束位置"
- 范围可以重叠（会自动去重）
- 示例：`"0-50,100-150,200-250"` 表示三个不连续的范围

## 完整参数说明

```
必需参数：
  input                输入JSON文件路径
  output               输出JSON文件路径
  --mode {all,continuous,range,multi-range}
                       替换模式

可选参数：
  --start START        起始位置（用于continuous和range模式，默认0）
  --length LENGTH      连续长度（用于continuous模式，默认10）
  --end END            结束位置（用于range模式，默认10）
  --ranges RANGES      多个范围（用于multi-range模式），格式如: "2-60,200-270,300-320,350-383"
  --min MIN            随机数最小值（默认0）
  --max MAX            随机数最大值（默认16383）
  --seed SEED          随机种子（用于可重复的随机结果）
```

## 快速测试

运行测试脚本来测试所有三种模式：

```bash
./test_randomize.sh
```

这将在 `data_test_output/` 目录中生成测试结果。

## 验证输出

运行验证脚本来检查输出结果：

```bash
python3 verify_output.py
```

这将显示：
- 每个文件中的token数量
- 原始文件和处理后文件的对比
- 变化的token数量和位置

## 高级用法示例

### 1. 使用固定随机种子（可重复结果）
```bash
python3 randomize_tokens.py \
    input.json output.json \
    --mode all \
    --seed 42
```

### 2. 自定义随机数范围
```bash
python3 randomize_tokens.py \
    input.json output.json \
    --mode all \
    --min 1000 \
    --max 15000
```

### 3. 多范围随机化的实际应用场景

**场景A：只随机化开头和结尾，保留中间部分**
```bash
python3 randomize_tokens.py \
    input.json output.json \
    --mode multi-range \
    --ranges "0-50,330-383"
```

**场景B：随机化多个特定区域（如不同的特征组）**
```bash
python3 randomize_tokens.py \
    input.json output.json \
    --mode multi-range \
    --ranges "0-95,96-191,192-287,288-383"
```

**场景C：保留关键位置，随机化其他所有位置**
```bash
# 假设位置100-110是关键特征，需要保留
# 那么随机化 0-99 和 111-383
python3 randomize_tokens.py \
    input.json output.json \
    --mode multi-range \
    --ranges "0-99,111-383"
```

**场景D：自定义任意组合**
```bash
# 你可以根据需要指定任意多个范围
python3 randomize_tokens.py \
    input.json output.json \
    --mode multi-range \
    --ranges "2-60,200-270,300-320,350-383"
```

### 4. 批量处理使用多范围模式
```bash
./batch_randomize.sh data/ output/ multi-range --ranges "2-60,200-270,300-320,350-383"
```

## 批量处理

如果需要处理多个文件，可以使用bash循环：

```bash
for file in data_test/*.json; do
    filename=$(basename "$file")
    python3 randomize_tokens.py \
        "$file" \
        "output/${filename%.json}_randomized.json" \
        --mode all
done
```

## 注意事项

1. 位置索引从0开始，范围是 [0, 383]
2. 在range模式中，end位置是包含的（inclusive）
3. 脚本会保持JSON文件的其他部分不变，只替换token中的数字
4. 输出文件会以UTF-8编码保存，带有缩进格式
5. 随机数范围是 [min, max]，包含两端

## 测试结果示例

运行 `./test_randomize.sh` 后的验证结果：

- **模式1（全部随机）**：384个token全部改变
- **模式2（连续随机，start=10, length=20）**：位置10-29的20个token改变
- **模式3（范围随机，start=50, end=100）**：位置50-100的51个token改变
- **模式4（多范围随机，ranges="2-60,200-270,300-320,350-383"）**：
  - 位置2-60：59个token改变
  - 位置200-270：71个token改变
  - 位置300-320：21个token改变
  - 位置350-383：34个token改变
  - 总计：185个token改变
