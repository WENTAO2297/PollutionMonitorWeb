import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pydeck as pdk
from datetime import datetime

# 引入工具
from weather_tool import get_shanghai_weather
# <--- 引入新的工具函数
from traffic_tool import get_landmarks_traffic


# ==========================================
# 1. 模型结构
# ==========================================
class AdvancedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdvancedBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ==========================================
# 2. 资源加载
# ==========================================
@st.cache_resource
def load_system():
    try:
        df = pd.read_csv('shanghai_traffic_simulation.csv')
        features = ['Traffic_Volume', 'Average_Speed', 'Temperature', 'Humidity', 'Hour']
        target = 'NOx_Emission'
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        scaler_X.fit(df[features].values)
        scaler_y.fit(df[[target]].values)

        input_dim = 5
        hidden_dim = 128
        output_dim = 1
        model = AdvancedBiLSTM(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load('BiLSTM_Best.pth', map_location=torch.device('cpu')))
        model.eval()
        return scaler_X, scaler_y, model
    except:
        return None, None, None


# ==========================================
# 3. 核心：精准点位数据生成
# ==========================================
def generate_xuhui_grid(traffic_data_dict, hour, temp, humid, scaler_X, scaler_y, model):
    """
    traffic_data_dict: 这是一个字典，包含了每个地标的真实速度 { "徐家汇": {speed:20...}, ... }
    """

    # 定义地标列表
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

    # 遍历每个点，去字典里查它自己的真实路况
    for name, lat, lon in landmarks:

        if traffic_data_dict and name in traffic_data_dict:
            # 命中真实数据！
            real_info = traffic_data_dict[name]
            local_speed = real_info['speed']
            congestion_factor = real_info['factor']
        else:
            # 没查到（或模拟模式），给个基于时间的模拟值
            is_peak = (8 <= hour <= 10) or (17 <= hour <= 19)
            local_speed = 25 if is_peak else 45
            congestion_factor = 2.0 if is_peak else 1.0

        # 估算流量 (流量 = 基准 * 拥堵系数)
        local_volume = 2000 * congestion_factor

        # AI 预测
        input_data = np.array([[local_volume, local_speed, temp, humid, hour]])
        input_scaled = scaler_X.transform(input_data)
        seq_input = torch.tensor(np.tile(input_scaled, (24, 1)), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = scaler_y.inverse_transform(model(seq_input).numpy())[0][0]
            pred = max(0, pred)

        # 颜色映射 (归一化到 0-150 范围)
        norm_val = min(pred / 150, 1.0)
        # 红(255,0,0) -> 绿(0,255,0) 的反向渐变：污染越重越红
        color = [255, int(255 * (1 - norm_val)), 0, 200]
        radius = 80 + pred * 4  # 排放越大圈越大

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

# 自动初始化
if 'init_done' not in st.session_state:
    with st.spinner('📡 连接北斗卫星与环境监测站...'):
        t, h, src_w = get_shanghai_weather()
        st.session_state.update({'temp': t, 'humid': h, 'w_src': src_w, 'init_done': True})

# --- 侧边栏 ---
with st.sidebar:
    st.title("🏙️ 监测控制台")
    st.caption(f"📍 区域：上海市徐汇区")

    if st.button("🔄 刷新全网数据"):
        # 清除缓存强制刷新
        get_landmarks_traffic.clear()
        st.session_state['init_done'] = False
        st.rerun()

    st.markdown("### 🌡️ 天气状况（实时）")
    c1, c2 = st.columns(2)
    c1.metric("气温", f"{st.session_state['temp']}°C")
    c2.metric("湿度", f"{st.session_state['humid']}%")
    st.caption(f"源: {st.session_state['w_src']}")

    st.markdown("### 🚦 交通数据流")
    st.info("已接入高德 API (Point-Level Mode)")
    st.caption("系统正在独立扫描每个监测点的周边路况...")

# --- 主界面 ---
st.title("徐汇区 NOx 排放全区预测 (实时)")
st.caption("基于Bi-LSTM 深度学习 | 高德交通大数据驱动 | GIS 空间分布")

if model and scaler_X:
    curr_hour = datetime.now().hour

    # 1. 现场去爬每个点的数据！
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

    # 调用新写的批量爬虫
    traffic_dict = get_landmarks_traffic(landmarks_coords)

    # 2. 生成全区分布
    df_xuhui = generate_xuhui_grid(
        traffic_dict, curr_hour,
        st.session_state['temp'], st.session_state['humid'],
        scaler_X, scaler_y, model
    )

    # 3. 寻找极值
    avg_nox = df_xuhui['nox'].mean()
    max_row = df_xuhui.loc[df_xuhui['nox'].idxmax()]
    min_row = df_xuhui.loc[df_xuhui['nox'].idxmin()]

    # 4. 顶部 KPI 卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据同步", datetime.now().strftime("%H:%M:%S"))
    c2.metric("全区平均排放", f"{avg_nox:.2f} mg", delta="正常" if avg_nox < 80 else "偏高", delta_color="inverse")

    # 这里展示差异化！最堵和最畅通的地方
    c3.metric("🔴 污染最多", f"{max_row['name']}", f"{max_row['nox']:.1f} mg")
    c4.metric("🟢 污染最少", f"{min_row['name']}", f"{min_row['nox']:.1f} mg")

    # 5. 地图
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

    view_state = pdk.ViewState(latitude=31.175, longitude=121.435, zoom=12.2, pitch=0)

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>{name}</b><br/>NOx预测: <b>{nox}</b> mg<br/>实测车速: {speed} km/h"}
    ))

    # 6. 数据表格
    with st.expander("查看各站点实时详情"):
        # 格式化一下表格，好看点
        show_df = df_xuhui[['name', 'speed', 'nox']].copy()
        show_df.columns = ['站点名称', '实时车速 (km/h)', 'NOx排放预测 (mg)']
        st.dataframe(show_df.style.background_gradient(subset=['NOx排放预测 (mg)'], cmap='OrRd'))

else:
    st.info("🔄 系统初始化中...")