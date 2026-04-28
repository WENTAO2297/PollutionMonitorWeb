import requests
import streamlit as st

# ==========================================
# 🔴 请替换为你自己申请的高德 Key
# ==========================================
AMAP_KEY = "096d3b310d1168ef856cc90fa9a3bb1c"


@st.cache_data(ttl=60)  # 💡 关键：设置60秒缓存，防止刷新太快耗尽高德免费额度
def get_landmarks_traffic(landmarks_list):
    """
    批量获取多个坐标点的实时路况
    输入: [("地名", lat, lon), ...]
    输出: { "地名": {speed: 30, status: 1.5}, ... }
    """
    if "你的高德" in AMAP_KEY:
        print("⚠️ 未配置高德Key，使用模拟数据")
        return None  # 返回 None 触发模拟逻辑

    print("📡 [API] 正在通过高德卫星查询各个站点路况...")
    results = {}

    try:
        for name, lat, lon in landmarks_list:
            # 查询圆形区域：中心点坐标，半径1000米
            location = f"{lon},{lat}"  # 高德要求经度在前
            url = f"https://restapi.amap.com/v3/traffic/status/circle?location={location}&radius=1000&key={AMAP_KEY}&extensions=all"

            resp = requests.get(url, timeout=2)
            data = resp.json()

            if data['status'] == '1' and 'trafficinfo' in data:
                info = data['trafficinfo']
                evaluation = info.get('evaluation', '未知')

                # 解析拥堵描述，转为系数
                if "畅通" in evaluation:
                    factor = 1.0
                elif "缓行" in evaluation:
                    factor = 1.5
                elif "拥堵" in evaluation:
                    factor = 2.0
                else:
                    factor = 2.5  # 严重拥堵或其他

                # 提取道路列表计算平均速度
                roads = info.get('roads', [])
                total_speed = 0
                count = 0
                for r in roads:
                    total_speed += float(r.get('speed', 0))
                    count += 1

                avg_speed = int(total_speed / count) if count > 0 else 40

                results[name] = {
                    "speed": avg_speed,
                    "factor": factor,
                    "desc": evaluation
                }
            else:
                # 查不到就给个默认值
                results[name] = {"speed": 40, "factor": 1.0, "desc": "无数据"}

        return results

    except Exception as e:
        print(f"⚠️ API连接错误: {e}")
        return None