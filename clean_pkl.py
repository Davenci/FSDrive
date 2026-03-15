import pickle
from nuscenes.nuscenes import NuScenes

# 1. 指向你刚才解压的 mini 数据集目录
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)

# 2. 获取这 10 个场景里所有合法的 Token
mini_tokens = set([sample['token'] for sample in nusc.sample])

# 3. 读取作者给的庞大完整的 pkl
with open('./create_data/cached_nuscenes_info_full.pkl', 'rb') as f:
    full_data = pickle.load(f)

# 4. 剔除多余部分：只保留在 mini_tokens 里的数据
if isinstance(full_data, list):
    mini_data = [item for item in full_data if item.get('token', item.get('sample_token', '')) in mini_tokens]
else:
    mini_data = {k: v for k, v in full_data.items() if k in mini_tokens}

# 5. 保存为你真正要用的 mini 版 pkl
with open('./create_data/cached_nuscenes_info.pkl', 'wb') as f:
    pickle.dump(mini_data, f)

print(f"剔除完成！保留了 {len(mini_data)} 条 mini 数据。")