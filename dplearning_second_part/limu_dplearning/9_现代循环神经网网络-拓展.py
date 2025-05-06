import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from utils.useful_func import *


## 深度循环神经网络
class RNN(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_layers, device):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.output_size = self.input_size = self.vocab_size = vocab_size
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.num_hiddens, num_layers=self.num_layers).to(device)
        self.state = self.begin_state
        self.linear = nn.Linear(self.num_hiddens, self.output_size).to(device)

    # input batchsize*numsteps
    def forward(self, inputs, state):
        inputs = inputs.T.to(dtype=torch.int64)
        x = F.one_hot(inputs, self.vocab_size).type(torch.float32)
        Y, state = self.lstm(x, state)

        # 这个地方需要reshape 因为最后的y是1维的 而这个result 是三维要变成2维的 才可以满足交叉熵损失函数的要求
        result = self.linear(Y.reshape(-1, Y.shape[-1]))
        return result, state

    # 初始化隐状态
    def begin_state(self, batch_size, device):
        return torch.zeros((1, batch_size, hidden_size), device=device), torch.zeros((1, batch_size, hidden_size),
                                                                                     device=device)

net=RNN(vocab_size=2, num_hiddens=2, num_layers=2, device='cpu')