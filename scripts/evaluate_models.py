import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'shanghai_traffic_simulation.csv'
OUTPUT_PATH = PROJECT_ROOT / 'assets' / 'model_comparison.png'

# ==========================================
# 1. 字体与高宽比设置 (学术规范)
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC'] 
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 2. 定义各个模型的网络结构 (根据报错日志精准反推还原)
# ==========================================

import torch.nn.functional as F  # 确保顶部有这个引用

# ==========================================
# 2. 定义各个模型的网络结构
# ==========================================

# (1) 引入注意力机制 (Temporal Attention)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        attn_weights = torch.tanh(self.attention(lstm_out))
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights

# (2) 最新版：BiLSTM-Attention 模型 (对应 models/bilstm_attention.pth)
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

# ... 下面的 BasicBiLSTM, BasicGRU, BasicLSTM 保持不变 ...

# (2) 基础版 Bi-LSTM (对应 Bi-LSTM_model.pth，2层，64节点，单FC层)
class BasicBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(BasicBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# (3) 基础版 GRU (对应 GRU_model.pth，2层，64节点，单FC层)
class BasicGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(BasicGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# (4) 基础版 LSTM (对应 LSTM_model.pth，2层，64节点，单FC层)
class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ==========================================
# 3. 数据加载与模型推理
# ==========================================
def run_all_models():
    print("正在加载数据集...")
    df = pd.read_csv(DATA_PATH)
    features = ['Traffic_Volume', 'Average_Speed', 'Temperature', 'Humidity', 'Hour']
    target = 'NOx_Emission'
    
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(df[features].values)
    scaler_y.fit(df[[target]].values)

    input_dim = 5
    output_dim = 1

    # 配置字典：分别使用对应的还原类实例化模型
    model_configs = {
        # 👇 名字改得更酷炫一点，并且实例化 BiLSTMAttention
        "BiLSTM-Attention模型": {"file": PROJECT_ROOT / "models" / "bilstm_attention.pth", "model": BiLSTMAttention(input_dim, 128, output_dim)},
        
        "Bi-LSTM模型":          {"file": PROJECT_ROOT / "legacy" / "Bi-LSTM_model.pth", "model": BasicBiLSTM(input_dim, 64, output_dim)},
        "GRU 模型":                 {"file": PROJECT_ROOT / "legacy" / "GRU_model.pth", "model": BasicGRU(input_dim, 64, output_dim)},
        "LSTM 模型":                {"file": PROJECT_ROOT / "legacy" / "LSTM_model.pth", "model": BasicLSTM(input_dim, 64, output_dim)}
    }
    # 取最后 200 个测试样本
    test_samples = 200
    df_test = df.tail(test_samples).reset_index(drop=True)
    true_values = df_test[target].values

    all_predictions = {}

    for name, config in model_configs.items():
        print(f"正在测试模型: {name} ({config['file']})...")
        model = config['model']
        try:
            model.load_state_dict(torch.load(config['file'], map_location=torch.device('cpu')))
            model.eval()
            
            preds = []
            with torch.no_grad():
                for i in range(len(df_test)):
                    row_features = df_test.loc[i, features].values.astype(float)
                    input_scaled = scaler_X.transform([row_features])
                    seq_input = torch.tensor(np.tile(input_scaled, (24, 1)), dtype=torch.float32).unsqueeze(0)
                    
                    pred = scaler_y.inverse_transform(model(seq_input).numpy())[0][0]
                    # 只有最佳优化版应用了彻底的非负物理约束，我们这里保留它们最原始的输出，看看老模型是不是会预测出负数污染
                    preds.append(pred)
            
            all_predictions[name] = preds
            print(f"✅ {name} 加载并推断成功！")
        except Exception as e:
            print(f"⚠️ 无法加载 {config['file']}，报错信息: {e}")
            all_predictions[name] = [0] * test_samples

    return true_values, all_predictions

# ==========================================
# 4. 绘制 2x2 四子图矩阵
# ==========================================
def plot_4_subplots(true_values, all_predictions):
    print("正在生成对比图表...")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), dpi=300)
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    
    for idx, (name, preds) in enumerate(all_predictions.items()):
        ax = axes[idx]
        # 真实值 (红色星号)
        ax.plot(true_values, color='red', marker='*', linestyle='-', linewidth=1, markersize=4, label='真实值 (True)')
        # 预测值 (空心圆圈)
        ax.plot(preds, color=colors[idx], marker='o', markerfacecolor='none', linestyle='-', linewidth=1, markersize=4, label=f'预测值 (Pred)')
        
        ax.set_title(f'{name} 预测结果对比', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('预测样本', fontsize=10)
        ax.set_ylabel('NOx 预测结果 (mg)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('徐汇区交通 NOx 排放多模型预测性能对比', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"🎯 绘图大功告成！已保存为 {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    y_true, y_preds_dict = run_all_models()
    plot_4_subplots(y_true, y_preds_dict)
