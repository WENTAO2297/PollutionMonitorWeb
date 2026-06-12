import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pydeck as pdk
from datetime import datetime
import pytz
import random
import torch.nn.functional as F
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.weather_service import get_shanghai_weather
from src.traffic_service import get_amap_api_key, get_landmarks_traffic


DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'shanghai_traffic_simulation.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'bilstm_attention.pth'

# ==========================================
# 1. 模型结构 (Attention 版本)
# ==========================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        attn_weights = torch.tanh(self.attention(lstm_out))
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = TemporalAttention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        attn_out, attn_weights = self.attention(lstm_out)
        out = self.relu(self.fc1(attn_out))
        out = self.fc2(out)
        return out

# ==========================================
# 2. 资源加载 (修复了缺失的Scaler代码)
# ==========================================
@st.cache_resource
def load_system():
    try:
        # 恢复了被删除的读取数据和归一化代码
        df = pd.read_csv(DATA_PATH)
        features = ['Traffic_Volume', 'Average_Speed', 'Temperature', 'Humidity', 'Hour']
        target = 'NOx_Emission'
        
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        scaler_X.fit(df[features].values)
        scaler_y.fit(df[[target]].values)

        input_dim = 5
        hidden_dim = 128
        output_dim = 1
        
        # 实例化带注意力的模型并加载权重
        model = BiLSTMAttention(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        return scaler_X, scaler_y, model
    except Exception as e:
        print(f"后台加载失败: {e}") 
        return None, None, None

# ==========================================
# 3. 核心：精准点位数据生成
# ==========================================
def generate_xuhui_grid(traffic_data_dict, hour, temp, humid, scaler_X, scaler_y, model):
    landmarks = [
        ("徐家汇商圈", 31.196, 121.436),
        ("上海体育场", 31.183, 121.442),
        ("上海南站", 31.153, 121.430),
        ("漕河泾开发区", 31.170, 121.397),
        ("徐汇滨江", 31.168, 121.465),
        ("衡山路", 31.205, 121.445),
        ("龙华寺", 31.176, 121.447),
        ("田林新村", 31.178, 121.417),
        ("华东理工", 31.143, 121.422),
        ("交大徐汇", 31.200, 121.430),
        ("美罗城", 31.192, 121.438),
        ("宜家家居", 31.172, 121.428)
    ]

    data_list = []

    for name, lat, lon in landmarks:
        if traffic_data_dict and name in traffic_data_dict and traffic_data_dict[name] is not None and traffic_data_dict[name]['speed'] is not None:
            real_info = traffic_data_dict[name]
            local_speed = real_info['speed']
            congestion_factor = real_info['factor']
            
            # 1. 引入地理特征基数 (给不同路口加上不同的流量权重，打破同质化)
            base_vol_map = {"徐家汇商圈": 3500, "上海南站": 4000, "衡山路": 1200, "徐汇滨江": 2500}
            base_vol = base_vol_map.get(name, 2000) # 默认基准 2000
            
            # 当前估算流量
            current_volume = base_vol * congestion_factor
            
            # 2. 构造 24 小时潮汐模拟序列 (绝不再用粗暴的完全复制)
            # 根据当前时间，倒推过去 24 小时的模拟特征
            seq_features = []
            for h_offset in range(24, 0, -1):
                # 模拟历史时间点
                hist_hour = (hour - h_offset) % 24
                
                # 模拟历史车速和流量的波动 (加入随机微扰和早晚高峰潮汐系数)
                # 假设早高峰8-10点，晚高峰17-19点，历史波动在 0.6 到 1.3 之间
                time_factor = 1.2 if hist_hour in [8, 9, 17, 18] else 0.8
                random_noise = random.uniform(0.9, 1.1)
                
                hist_volume = current_volume * time_factor * random_noise
                hist_speed = local_speed * (1 / time_factor) * random_noise # 流量大则速度慢
                hist_temp = temp - random.uniform(0, 3) # 过去可能稍微冷一点
                
                seq_features.append([hist_volume, hist_speed, hist_temp, humid, hist_hour])
            
            # 转换为 numpy 数组并归一化
            input_data = np.array(seq_features)
            input_scaled = scaler_X.transform(input_data)
            
            # 组成 3D 张量 (1, 24, 5)
            seq_input = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
            
            # --- 以下推理部分保持不变 ---
            with torch.no_grad():
                pred = scaler_y.inverse_transform(model(seq_input).numpy())[0][0]
                pred = max(0, pred)
                
            norm_val = min(pred / 150, 1.0)
            color = [255, int(255 * (1 - norm_val)), 0, 200]
            radius = 80 + pred * 4 

            data_list.append({
                "name": name, "lat": lat, "lon": lon,
                "nox": float(pred), "speed": int(local_speed),
                "humid": humid, "color": color, "radius": radius
            })
            
    return pd.DataFrame(data_list)

# ==========================================
# 4. 页面主程序
# ==========================================
st.set_page_config(page_title="徐汇区排放数字孪生", page_icon="🏙️", layout="wide")

scaler_X, scaler_y, model = load_system()

if 'init_done' not in st.session_state:
    with st.spinner('📡 连接气象与环境监测站...'):
        t, h, src_w = get_shanghai_weather()
        st.session_state.update({'temp': t, 'humid': h, 'w_src': src_w, 'init_done': True})

if 'map_center_lat' not in st.session_state:
    st.session_state['map_center_lat'] = 31.175

with st.sidebar:
    st.title("🏙️ 监测控制台")
    st.caption(f"📍 区域：上海市徐汇区")

    if st.button("🔄 刷新全网数据"):
        get_landmarks_traffic.clear()
        st.session_state['init_done'] = False
        st.rerun()
        
    if st.button("📍 视角复位"):
        st.session_state['map_center_lat'] = 31.175 + random.uniform(-0.00001, 0.00001)

    st.markdown("### 🌡️ 天气状况（实时）")
    c1, c2 = st.columns(2)
    c1.metric("气温", f"{st.session_state['temp']}°C")
    c2.metric("湿度", f"{st.session_state['humid']}%")
    st.caption(f"源: {st.session_state['w_src']}")

    st.markdown("### 🚦 交通数据流")
    if get_amap_api_key():
        st.info("已接入高德 API (Point-Level Mode)")
        st.caption("系统正在独立扫描每个监测点的周边路况...")
    else:
        st.warning("未配置高德 API Key，当前使用离线模拟交通数据")

st.title("徐汇区 NOx 排放全区预测 (实时)")
st.caption("基于 BiLSTM-Attention 深度学习 | 高德交通大数据驱动 | GIS 空间分布")

if model and scaler_X:
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    curr_time = datetime.now(shanghai_tz)
    curr_hour = curr_time.hour

    landmarks_coords = [
        ("徐家汇商圈", 31.196, 121.436),
        ("上海体育场", 31.183, 121.442),
        ("上海南站", 31.153, 121.430),
        ("漕河泾开发区", 31.170, 121.397),
        ("徐汇滨江", 31.168, 121.465),
        ("衡山路", 31.205, 121.445),
        ("龙华寺", 31.176, 121.447),
        ("田林新村", 31.178, 121.417),
        ("华东理工", 31.143, 121.422),
        ("交大徐汇", 31.200, 121.430),
        ("美罗城", 31.192, 121.438),
        ("宜家家居", 31.172, 121.428)
    ]

    traffic_dict = get_landmarks_traffic(landmarks_coords)
    
    if get_amap_api_key():
        print("\n[高德 API] 获取到的交通数据如下：")
    else:
        print("\n[离线模式] 生成的模拟交通数据如下：")
    print(traffic_dict)

    df_xuhui = generate_xuhui_grid(
        traffic_dict, curr_hour,
        st.session_state['temp'], st.session_state['humid'],
        scaler_X, scaler_y, model
    )

    avg_nox = df_xuhui['nox'].mean()
    max_row = df_xuhui.loc[df_xuhui['nox'].idxmax()]
    min_row = df_xuhui.loc[df_xuhui['nox'].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据同步", curr_time.strftime("%H:%M:%S"))
    c2.metric("全区平均排放", f"{avg_nox:.2f} mg", delta="正常" if avg_nox < 80 else "偏高", delta_color="inverse")
    c3.metric("🔴 污染最多", f"{max_row['name']}", f"{max_row['nox']:.1f} mg")
    c4.metric("🟢 污染最少", f"{min_row['name']}", f"{min_row['nox']:.1f} mg")

    st.subheader("📍 区域排放热力图")

    layer = pdk.Layer(
        "ScatterplotLayer",
        df_xuhui,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.8,
        filled=True,
        radius_min_pixels=8,
        radius_max_pixels=60,
    )

    view_state = pdk.ViewState(
        latitude=st.session_state['map_center_lat'], 
        longitude=121.435, 
        zoom=12.2, 
        pitch=0,
        min_zoom=11.5,
        max_zoom=16
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>{name}</b><br/>NOx预测: <b>{nox}</b> mg<br/>实测车速: {speed} km/h"}
    ))

    with st.expander("查看各站点实时详情"):
        show_df = df_xuhui[['name', 'speed', 'nox']].copy()
        show_df.columns = ['站点名称', '实时车速 (km/h)', 'NOx排放预测 (mg)']
        st.dataframe(show_df.style.background_gradient(subset=['NOx排放预测 (mg)'], cmap='OrRd'))

else:
    st.error("后台组件加载失败，请检查终端日志中的具体报错原因。")
