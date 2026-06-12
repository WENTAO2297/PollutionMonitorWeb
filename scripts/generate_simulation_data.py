import pandas as pd
import numpy as np
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_DATA_PATH = PROJECT_ROOT / 'data' / 'sample' / 'vehicle_emissions_2026.csv'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'shanghai_traffic_simulation.csv'

# ==========================================
# 1. 读取并清洗你的数据集
# ==========================================
print(f"正在读取车辆数据: {VEHICLE_DATA_PATH} ...")
try:
    df_gas = pd.read_csv(VEHICLE_DATA_PATH)
except FileNotFoundError:
    print(f"错误: 找不到数据文件 {VEHICLE_DATA_PATH}！")
    exit()

# 清洗数据：确保数值列没有空值
df_gas['Smog rating'] = pd.to_numeric(df_gas['Smog rating'], errors='coerce').fillna(5)
df_gas['CO2 emissions (g/km)'] = pd.to_numeric(df_gas['CO2 emissions (g/km)'], errors='coerce').fillna(200)

print(f"燃油/混动车数据加载完成，共 {len(df_gas)} 款车型。")


# ==========================================
# 2. 定义仿真逻辑
# ==========================================

def generate_shanghai_data(days=60):
    records = []

    # 上海交通参数设定
    prob_ev = 0.35  # 新能源渗透率 35% (代码自动生成这部分数据)
    prob_gas = 0.65  # 燃油车比例 (从车辆排放样本中采样)

    print(f"开始生成 {days} 天的仿真交通数据...")

    for day in range(days):
        for hour in range(24):
            # --- A. 模拟宏观交通流 (Traffic Flow) ---
            # 逻辑：早晚高峰车多速慢，深夜车少速快
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰期
                volume = int(np.random.normal(3500, 400))  # 流量大
                speed = np.random.normal(20, 5)  # 车速极慢 (拥堵)
                congestion_factor = 1.6  # 拥堵导致排放激增
            elif 10 <= hour <= 16:  # 平峰期
                volume = int(np.random.normal(1800, 300))
                speed = np.random.normal(40, 8)
                congestion_factor = 1.1
            elif 0 <= hour <= 5:  # 深夜
                volume = int(np.random.normal(300, 100))
                speed = np.random.normal(65, 10)
                congestion_factor = 1.0  # 畅通
            else:  # 过渡期
                volume = int(np.random.normal(1200, 200))
                speed = np.random.normal(50, 8)
                congestion_factor = 1.0

            # 修正车速不能小于 1
            speed = max(1.0, speed)

            # --- B. 计算该小时的总排放 (Bottom-Up Simulation) ---

            # 1. 计算不同车型的数量
            n_ev = int(volume * prob_ev)
            n_gas = volume - n_ev

            total_nox = 0
            total_co2 = 0

            # 2. 燃油车排放计算 (基于车辆排放样本)
            if n_gas > 0:
                # 随机抽取 n_gas 辆车
                sample = df_gas.sample(n=n_gas, replace=True)

                # 计算 CO2: 基础排放 * 拥堵系数
                batch_co2 = sample['CO2 emissions (g/km)'].sum() * congestion_factor
                total_co2 += batch_co2

                # 计算 NOx (氮氧化物): 利用 Smog Rating 反推
                # Rating 越高越清洁。假设 NOx 与 (11 - Rating) 成正比
                # 系数 8.0 是经验值，用于将等级转换为 mg/km 近似值
                ratings = sample['Smog rating'].values
                batch_nox = np.sum((11 - ratings) * 8.0) * congestion_factor
                total_nox += batch_nox

            # 3. 电车排放计算
            # 纯电车行驶排放为 0 (无需计算)

            # --- C. 环境数据 (增加一些随机天气特征) ---
            # 模拟气温：白天热晚上冷
            base_temp = 22 - 5 * np.cos(hour * np.pi / 12)
            temp = base_temp + np.random.normal(0, 2)
            # 模拟湿度
            humidity = 60 + 10 * np.sin(hour * np.pi / 12) + np.random.normal(0, 5)

            # --- D. 记录数据 ---
            records.append({
                'Hour': hour,
                'Traffic_Volume': volume,
                'Average_Speed': round(speed, 1),
                'Temperature': round(temp, 1),
                'Humidity': round(humidity, 1),
                'NOx_Emission': round(total_nox / 1000, 2),  # 换算为克(g) 或 千克
                # CO2 仅作参考，不一定作为特征
            })

    return pd.DataFrame(records)


# 执行生成
df_sim = generate_shanghai_data(days=90)  # 生成 3 个月的数据，保证训练量充足
print(f"数据生成完成! 形状: {df_sim.shape}")
print(df_sim.head())

# 保存文件
df_sim.to_csv(OUTPUT_PATH, index=False)
print(f">>> 文件已保存为: {OUTPUT_PATH} <<<")
