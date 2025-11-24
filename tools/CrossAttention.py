import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim):
        super(CrossAttention, self).__init__()

        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)

    def forward(self, input_a, input_b):
        # 线性映射
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)
        y = mapped_b.transpose(1, 2)

        # 计算注意力权重
        scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))  # (batch_size, seq_len_a, seq_len_b)
        attentions_a = torch.softmax(scores, dim=-1)  # 在维度2上进行softmax，归一化为注意力权重 (batch_size, seq_len_a, seq_len_b)
        attentions_b = torch.softmax(scores.transpose(1, 2),
                                     dim=-1)  # 在维度1上进行softmax，归一化为注意力权重 (batch_size, seq_len_b, seq_len_a)

        # 使用注意力权重来调整输入表示
        output_a = torch.matmul(attentions_b, input_b)  # (batch_size, seq_len_a, input_dim_b)
        output_b = torch.matmul(attentions_a.transpose(1, 2), input_a)  # (batch_size, seq_len_b, input_dim_a)

        return output_a, output_b


if __name__ == '__main__':
    # 准备数据
    input_a = torch.randn(16, 36, 192)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
    input_b = torch.randn(16, 192, 36)  # 输入序列B，大小为(batch_size, seq_len_b, input_dim_b)
    # 定义模型
    input_dim_a = input_a.shape[-1]
    input_dim_b = input_b.shape[-1]
    hidden_dim = 64
    cross_attention = CrossAttention(input_dim_a, input_dim_b, hidden_dim)

    # 前向传播
    output_a, output_b = cross_attention(input_a, input_b)
    print("Adjusted output A:\n", output_a)
    print("Adjusted output B:\n", output_b)
