import json

# 加载数据
with open('./LLaMA-Factory/results.jsonl', 'r', encoding='utf-8') as f:
    results = [json.loads(line) for line in f]

with open('./LLaMA-Factory/data/val_cot_motion.json', 'r', encoding='utf-8') as f:
    val_cot = json.load(f)

# 只取前100条
val_cot = val_cot[:100]

# 创建新的eval_traj字典，使用完整的label内容
eval_traj = {}

for i, item in enumerate(val_cot):
    token_id = item['id']
    label = results[i]['label']

    # 原样复制完整的label内容
    eval_traj[token_id] = label

# 保存结果
with open('./LLaMA-Factory/eval_traj.json', 'w', encoding='utf-8') as f:
    json.dump(eval_traj, f, indent=4, ensure_ascii=False)

print(f"Successfully updated eval_traj.json with {len(eval_traj)} entries")
