import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入我们自己写的模块
from data_process import load_and_process_data
from LSTM import AttentionLSTM

# ================= 配置参数 =================
FILE_PATH = 'AirQualityUCI.csv'  # 确保你的数据文件在同一目录下
TIME_STEP = 24  # 时间步长
HIDDEN_SIZE = 128  # LSTM隐藏层大小
LEARNING_RATE = 0.005
EPOCHS = 300  # 训练轮数


# ===========================================

def main():
    # 1. 加载数据
    try:
        X_train, y_train, X_test, y_test, scaler_y = load_and_process_data(FILE_PATH, TIME_STEP)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{FILE_PATH}'，请确保文件在当前目录下。")
        return

    # 2. 设置设备 (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 转为 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 3. 初始化模型
    input_dim = X_train.shape[2]  # 特征数量 (Traffic, Speed, T, RH) -> 4
    output_dim = 1  # 预测 NOx

    model = AttentionLSTM(input_dim, HIDDEN_SIZE, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练循环
    print("\n开始训练模型...")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}')

    # 5. 测试与评估
    print("\n正在评估...")
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).cpu().numpy()
        y_true = y_test_t.cpu().numpy()

    # 反归一化 (变回真实数值)
    test_preds_inv = scaler_y.inverse_transform(test_preds)
    y_true_inv = scaler_y.inverse_transform(y_true)

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true_inv, test_preds_inv))
    mae = mean_absolute_error(y_true_inv, test_preds_inv)
    r2 = r2_score(y_true_inv, test_preds_inv)

    print("-" * 30)
    print(f"评估结果 (NOx):")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print("-" * 30)

    # 6. 保存模型 (可选)
    torch.save(model.state_dict(), 'traffic_emission_model.pth')
    print("模型已保存为 traffic_emission_model.pth")

    # 7. 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_inv[:200], label='True Value', color='blue')
    plt.plot(test_preds_inv[:200], label='Prediction', color='red', linestyle='--')
    plt.title(f'Emission Prediction Results - R2: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('NOx Concentration')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()