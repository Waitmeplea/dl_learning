import torch
import torch.nn as nn
import sys
from utils.useful_func import *


##信息流的两阶段控制：筛选阶段（RtRt​）：决定历史信息如何参与候选状态计算。
# 混合阶段（）：决定最终状态的更新比例。
# 参数共享：在候选状态中复用，减少参数量。
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(shape, device=device) * 0.01

    def three():
        return normal((num_inputs, num_hiddens)), \
            normal((num_hiddens, num_hiddens)), \
            torch.zeros(num_hiddens, device=device)

    W_xr, W_hr, b_r = three()
    W_xz, W_hz, b_z = three()
    W_xh, W_hh, b_h = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros(batch_size, num_hiddens, device=device),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + (R * torch.matmul(H, W_hh) + b_h))
        H = Z * H + (1 - Z) * H_tilda

        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn
                 ):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, inputs, state):
        # 转置的目的是 为了能够直接通过最外层的维度直接循环 以训练序列数据
        ## 因为输入一般是(batch_size,num_steps) 所以需要转置
        X = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


class RNN(nn.Module):
    ##
    def __init__(self, vocab_size, num_hiddens, device):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.output_size = self.input_size = self.vocab_size = vocab_size
        self.device = device
        self.gru = nn.GRU(self.input_size, self.num_hiddens).to(device)
        self.state = self.begin_state
        self.linear = nn.Linear(self.num_hiddens, self.output_size).to(device)

    # input batchsize*numsteps
    def forward(self, inputs, state):
        inputs = inputs.T.to(dtype=torch.int64)
        x = F.one_hot(inputs, self.vocab_size).type(torch.float32)
        Y, state = self.gru(x, state)

        # 这个地方需要reshape 因为最后的y是1维的 而这个result 是三维要变成2维的 才可以满足交叉熵损失函数的要求
        result = self.linear(Y.reshape(-1,Y.shape[-1]))
        return result, state

    # 初始化隐状态
    def begin_state(self, batch_size, device):
        return torch.zeros((1, batch_size, self.num_hiddens), device=self.device)


if __name__ == '__main__':
    batch_size = 32
    num_steps = 35
    date_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=False)
    vocab_size, num_hiddens, device = len(vocab), 256, 'cpu'
    num_epochs, lr = 500, 1
    # model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
    #                         init_gru_state, gru)

    model = RNN(28, 256, 'cpu')
    train_ch8(model, date_iter, vocab, lr, num_epochs, device)
