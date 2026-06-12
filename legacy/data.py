import pandas as pd
import numpy as np
import random

# 1. 读取数据
print("正在读取车辆数据...")
# 注意：文件名需要和你上传的一致
df_ev = pd.read_csv('纯电.csv')
df_phev = pd.read_csv('插电混合.csv')
df_gas = pd.read_csv('燃油.csv')


# 2. 数据清洗函数
def clean_data(df, type_name):
    # 填充空值，防止报错
    df = df.fillna(0)
    # 确保有 Smog rating 列，如果没有（如纯电），设为最高分 10（最清洁）
    if 'Smog rating' not in df.columns:
        df['Smog rating'] = 10

        # 确保 Smog rating 是数字
    df['Smog rating'] = pd.to_numeric(df['Smog rating'], errors='coerce').fillna(5)

    # 提取 CO2
    if 'CO2 emissions (g/km)' in df.columns:
        df['co2_g_km'] = pd.to_numeric(df['CO2 emissions (g/km)'], errors='coerce').fillna(0)
    else:
        df['co2_g_km'] = 0

    return df


df_ev = clean_data(df_ev, 'EV')
df_phev = clean_data(df_phev, 'PHEV')
df_gas = clean_data(df_gas, 'Gas')

print(f"数据加载完成: 纯电 {len(df_ev)} 款, 插混 {len(df_phev)} 款, 燃油 {len(df_gas)} 款")


# 3. 定义 NOx 估算逻辑
# Smog Rating 通常是 1-10，10 最清洁。
# 我们假设 NOx (mg/km) 与 (11 - rating) 成正比。
# 现代汽油车 NOx 大约在 10-60 mg/km 之间。
def estimate_nox(smog_rating, fuel_type):
    if fuel_type == 'EV':
        return 0
    # 评分越低，污染越高。基准系数设为 8.0 (这是一个经验估算值)
    pollution_level = 11 - smog_rating
    noise = np.random.normal(0, 1.0)  # 加一点随机波动
    nox_val = pollution_level * 5.0 + noise
    return max(0, nox_val)  # 保证不小于0


# 4. 生成仿真交通数据
def generate_shanghai_data(days=30):
    records = []

    # 上海特征：新能源渗透率高 (假设 35%)
    prob_ev = 0.20
    prob_phev = 0.15
    prob_gas = 0.65

    for day in range(days):
        for hour in range(24):
            # === A. 模拟交通流 (Traffic Flow) ===
            # 早高峰 8-9点，晚高峰 17-19点
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                traffic_pressure = np.random.uniform(0.8, 1.0)  # 拥堵系数
                volume = int(np.random.normal(3000, 500))  # 车流量大
                speed = np.random.normal(25, 5)  # 车速慢 (25km/h)
            elif 10 <= hour <= 16:
                traffic_pressure = np.random.uniform(0.4, 0.6)
                volume = int(np.random.normal(1500, 300))
                speed = np.random.normal(45, 10)
            elif 0 <= hour <= 5:
                traffic_pressure = np.random.uniform(0.0, 0.1)
                volume = int(np.random.normal(200, 50))
                speed = np.random.normal(65, 10)  # 深夜车速快
            else:
                traffic_pressure = np.random.uniform(0.3, 0.5)
                volume = int(np.random.normal(1000, 200))
                speed = np.random.normal(50, 8)

            # === B. 计算该小时的总排放 (Bottom-Up) ===
            total_co2 = 0
            total_nox = 0

            # 快速算法：不用循环几千次，直接按比例采样
            # 1. 确定每种车的数量
            n_ev = int(volume * prob_ev)
            n_phev = int(volume * prob_phev)
            n_gas = volume - n_ev - n_phev

            # 2. 从真实数据集中随机抽取车辆样本
            # 燃油车排放
            if n_gas > 0:
                sample = df_gas.sample(n=n_gas, replace=True)
                # 车速低时(拥堵)，单位距离排放会增加 (简化模拟：乘以拥堵因子)
                congestion_factor = 1.5 if speed < 30 else 1.0

                total_co2 += sample['co2_g_km'].sum() * congestion_factor
                # 计算 NOx
                nox_ratings = sample['Smog rating'].values
                # 向量化计算 NOx
                total_nox += np.sum((11 - nox_ratings) * 5.0) * congestion_factor

            # 插混车排放 (假设一半用电一半用油)
            if n_phev > 0:
                sample = df_phev.sample(n=n_phev, replace=True)
                congestion_factor = 1.2 if speed < 30 else 1.0
                # 假设 50% 里程是混动模式产生排放
                total_co2 += (sample['co2_g_km'].sum() * 0.5) * congestion_factor
                nox_ratings = sample['Smog rating'].values
                total_nox += (np.sum((11 - nox_ratings) * 5.0) * 0.5) * congestion_factor

            # 纯电车排放 = 0 (本地排放)

            # === C. 整理数据 ===
            # 将“g/km”转换为“该小时路段总排放 (g)”
            # 假设路段长度 1 km

            # 添加一些环境噪声 (温度、湿度)
            temp = 20 + 5 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 2)
            humidity = 60 + 10 * np.cos(hour * np.pi / 12) + np.random.normal(0, 5)

            records.append({
                'Date': f'Day_{day + 1}',
                'Hour': hour,
                'Traffic_Volume': volume,
                'Average_Speed': round(speed, 1),
                'Temperature': round(temp, 1),
                'Humidity': round(humidity, 1),
                'Fleet_EV_Ratio': prob_ev,
                'NOx_Emission': round(total_nox, 2),  # 这是我们的 Target
                'CO2_Emission': round(total_co2, 2)  # 这是一个辅助特征
            })

    return pd.DataFrame(records)


# 5. 执行生成
print("正在基于真实车型生成上海交通仿真数据...")
df_sim = generate_shanghai_data(days=60)  # 生成 60 天的数据
print(f"生成完成！数据形状: {df_sim.shape}")
print(df_sim.head())

# 6. 保存
df_sim.to_csv('shanghai_traffic.csv', index=False)
print("已保存为 'shanghai_traffic.csv'")