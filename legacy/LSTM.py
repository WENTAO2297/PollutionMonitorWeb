import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    注意力机制层
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output):
        # rnn_output: [batch, seq_len, hidden_size]
        attn_weights = F.softmax(self.attn(rnn_output), dim=1)
        context_vector = torch.sum(attn_weights * rnn_output, dim=1)
        return context_vector, attn_weights

class AttentionLSTM(nn.Module):
    """
    带有注意力机制的LSTM模型
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        context_vector, _ = self.attention(out)
        prediction = self.fc(context_vector)
        return prediction