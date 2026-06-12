import requests
import random
from datetime import datetime


def get_shanghai_weather():
    """
    获取上海实时天气
    策略 A: Open-Meteo (速度快，科学数据)
    策略 B: wttr.in (备用)
    策略 C: 模拟数据 (保底)
    """
    print("正在尝试获取上海实时天气...")

    # --- 策略 A: Open-Meteo API (基于经纬度) ---
    try:
        # 上海坐标: 31.23N, 121.47E
        # 请求内容: 温度(2米高), 相对湿度(2米高)
        url_a = "https://api.open-meteo.com/v1/forecast?latitude=31.23&longitude=121.47&current=temperature_2m,relative_humidity_2m&timezone=Asia%2FShanghai"

        # 增加超时时间到 5 秒
        response = requests.get(url_a, timeout=5)

        if response.status_code == 200:
            data = response.json()
            current = data['current']
            temp = int(current['temperature_2m'])
            humidity = int(current['relative_humidity_2m'])

            print(f"✅ [OpenMeteo] 获取成功: {temp}°C, 湿度 {humidity}%")
            return temp, humidity, "Open-Meteo 实时数据"

    except Exception as e:
        print(f"⚠️ 策略 A 失败: {e}")

    # --- 策略 B: wttr.in (备用) ---
    try:
        url_b = "https://wttr.in/Shanghai?format=j1"
        response = requests.get(url_b, timeout=5)  # 增加到 5 秒

        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            temp = int(current['temp_C'])
            humidity = int(current['humidity'])

            print(f"✅ [wttr.in] 获取成功: {temp}°C, 湿度 {humidity}%")
            return temp, humidity, "wttr.in 实时数据"

    except Exception as e:
        print(f"⚠️ 策略 B 失败: {e}")

    # --- 策略 C: 模拟数据  ---
    print("⚠️ 所有网络接口均超时，切换至本地模拟模式...")
    hour = datetime.now().hour

    # 更加逼真的模拟算法
    # 模拟温度：白天(14点)热，晚上冷
    base_temp = 20 - 5 * abs(hour - 14) / 12
    mock_temp = int(base_temp + random.randint(-1, 3))

    # 模拟湿度：晚上湿度大，白天湿度小
    base_humid = 60 + 10 * abs(hour - 14) / 12
    mock_humidity = int(base_humid + random.randint(-5, 5))

    # 限制范围
    mock_humidity = max(20, min(100, mock_humidity))

    return mock_temp, mock_humidity, "离线模拟数据 (演示模式)"


if __name__ == "__main__":
    t, h, status = get_shanghai_weather()
    print(f"\n结果: 温度={t}, 湿度={h}, 来源={status}")