import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'shanghai_traffic_simulation.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'bilstm_attention.pth'

# ==========================================
# 1. 独立的时间注意力机制层 (Temporal Attention)
# ==========================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        # 定义一个线性层来计算注意力得分
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out 维度: (batch_size, sequence_length, hidden_dim)
        
        # 计算每个时间步的得分
        attn_weights = torch.tanh(self.attention(lstm_out)) # 维度: (batch_size, sequence_length, 1)
        
        # 使用 Softmax 归一化，得到真正的权重 (概率和为1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 将权重与原来的 LSTM 输出相乘，得到加权后的上下文向量
        # (batch_size, sequence_length, 1) * (batch_size, sequence_length, hidden_dim)
        context = torch.sum(attn_weights * lstm_out, dim=1) # 维度: (batch_size, hidden_dim)
        
        return context, attn_weights

# ==========================================
# 2. 全新升级：BiLSTM-Attention 模型
# ==========================================
class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(BiLSTMAttention, self).__init__()
        # 底层依然是强大的 3 层双向 LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        
        # 注意：双向 LSTM 的输出维度是 hidden_size * 2
        self.attention = TemporalAttention(hidden_size * 2)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. LSTM 提取时序特征
        lstm_out, _ = self.lstm(x) 
        # lstm_out 包含了所有 24 个时间步的隐藏状态
        
        # 2. 引入 Attention 机制分配权重
        # 抛弃原来粗暴的 `out[:, -1, :]` (只取最后一个时刻)
        # 改为使用 Attention 加权融合所有的时刻
        attn_out, attn_weights = self.attention(lstm_out)
        
        # 3. 经过非线性全连接层输出
        out = self.relu(self.fc1(attn_out))
        out = self.fc2(out)
        
        return out


# ==========================================
# 2. 训练逻辑
# ==========================================
def create_sequences(data, target, seq_length=24):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(target[i + seq_length])
    return np.array(xs), np.array(ys)


def main():
    print(">>> 读取数据...")
    try:
        df = pd.read_csv(DATA_PATH)
    except:
        print("❌ 找不到数据文件！")
        return

    features = ['Traffic_Volume', 'Average_Speed', 'Temperature', 'Humidity', 'Hour']
    target = 'NOx_Emission'

    # 归一化
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df[features].values)
    y_scaled = scaler_y.fit_transform(df[[target]].values)

    SEQ_LEN = 24
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32)
    )
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 模型初始化
    input_dim = 5
    hidden_dim = 128
    output_dim = 1
    # 👇 把 AdvancedBiLSTM 换成 BiLSTMAttention
    model = BiLSTMAttention(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)

    # 稍微降低学习率，防止震荡
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    print(f"\n🚀 开始训练 (Linear Output版)...")
    model.train()

    for epoch in range(100):  # 100 epoch 足够了
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"   Epoch [{epoch + 1}/100] | Loss: {avg_loss:.6f}")

    # 简单自测一下
    model.eval()
    with torch.no_grad():
        test_in = torch.tensor(X_seq[:5], dtype=torch.float32)
        test_out = model(test_in)
        real_val = scaler_y.inverse_transform(test_out.numpy())
        print(f"\n🔎 自测前5个样本预测值: {real_val.flatten().round(2)}")
        if np.all(real_val == 0):
            print("⚠️ 警告：预测值依然全为0，可能需要重新生成数据！")
        else:
            print("✅ 预测值正常，不是全0了！")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ 模型已保存为 {MODEL_PATH}")


if __name__ == "__main__":
    main()
