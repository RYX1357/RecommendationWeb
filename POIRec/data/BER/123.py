import pandas as pd
from datetime import datetime

# 读取数据
df = pd.read_csv('C:\\Users\\Ruany\\Desktop\\papercode\\H3GNN-main\\data_preprocess\\yongliu_gowalla_data\\Berlin\\check_in_berlin_user_in_friend.csv')

# 将datetime列转换为datetime类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 提取小时
df['hour'] = df['datetime'].dt.hour

# 定义时间段函数
def get_time_slot(hour):
    if 22 <= hour or hour < 5:
        return 'night'
    elif 5 <= hour < 10:
        return 'morning'
    elif 10 <= hour < 15:
        return 'noon'
    else:
        return 'evening'

# 应用时间段分类
df['time_slot'] = df['hour'].apply(get_time_slot)

# 初始化结果字典
slot_index = {'night': 0, 'morning': 1, 'noon': 2, 'evening': 3}
place_stats = {}

# 聚合统计
for _, row in df.iterrows():
    pid = row['placeid']
    slot = row['time_slot']
    if pid not in place_stats:
        place_stats[pid] = [0, 0, 0, 0]
    place_stats[pid][slot_index[slot]] += 1

# 转换为DataFrame
result_df = pd.DataFrame([
    {'placeid': pid, 'night': v[0], 'morning': v[1], 'noon': v[2], 'evening': v[3]}
    for pid, v in place_stats.items()
])

# 保存原始统计结果
result_df.to_csv('place_time_distribution.csv', index=False)

# ✅ 添加归一化：将每一行 night~evening 四列做归一化处理
time_cols = ['night', 'morning', 'noon', 'evening']
result_df_norm = result_df.copy()
result_df_norm[time_cols] = result_df_norm[time_cols].div(result_df_norm[time_cols].sum(axis=1), axis=0)

# 保存归一化结果
result_df_norm.to_csv('place_time_distribution_normalized.csv', index=False)

print("✅ 统计完成！已生成两个文件：")
print("👉 原始统计结果：place_time_distribution.csv")
print("👉 归一化结果：place_time_distribution_normalized.csv")

