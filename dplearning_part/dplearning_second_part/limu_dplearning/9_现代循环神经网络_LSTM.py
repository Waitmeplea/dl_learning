import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from utils.useful_func import *


def get_lstm_params(vocab_size, hidden_size, device):
    # 输入输出与词表大小一样
    input_size = output_size = vocab_size

    def normal(shape):
        return torch.randn(shape, device=device) * 0.01

    def three():
        return normal([input_size, hidden_size]), normal([hidden_size, hidden_size]), torch.zeros([hidden_size],
                                                                                                  device=device)

    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()
    W_output = normal([hidden_size, output_size])
    b_output = torch.zeros([output_size], device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_hf, b_o, W_xc, W_hc, b_c, W_output, b_output]
    for param in params:
        param.requires_grad_(True)
    return params


# 批量训练（Batch Processing）：
# 现代深度学习框架（如PyTorch、TensorFlow）通过同时处理多个样本（一个batch）来加速训练。隐状态需要为每个样本独立维护其时间步的隐藏信息，因此形状必须包含 batch_size。
def init_lstm_state(batch_size, hidden_size, device):
    return torch.zeros((batch_size, hidden_size), device=device), torch.zeros((batch_size, hidden_size),
                                                                              device=device)


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_output, b_output = params
    Ht, Ct = state

    output = []
    for X in inputs:
        It = F.sigmoid(torch.mm(X, W_xi) + torch.mm(Ht, W_hi) + b_i)
        Ft = F.sigmoid(torch.mm(X, W_xf) + torch.mm(Ht, W_hf) + b_f)
        Ot = F.sigmoid(torch.mm(X, W_xo) + torch.mm(Ht, W_ho) + b_o)
        C_tilda = torch.tanh(torch.mm(X, W_xc) + torch.mm(Ht, W_hc) + b_c)
        Ct = Ft * Ct + It * C_tilda
        Ht = Ot * torch.tanh(Ct)
        Y = torch.mm(Ht, W_output) + b_output
        output.append(Y)
    return torch.cat(output, dim=0), (Ht, Ct)


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn
                 ):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, inputs, state):
        # 转置的目的是 为了能够直接通过最外层的维度直接循环 以训练序列数据
        X = F.one_hot(inputs.T.long(), self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

###################################### 使用 pytorch框架的简洁实现
class RNN(nn.Module):
    ##
    def __init__(self, vocab_size, num_hiddens, device):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.output_size = self.input_size = self.vocab_size = vocab_size
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.num_hiddens).to(device)
        self.state = self.begin_state
        self.linear = nn.Linear(self.num_hiddens, self.output_size).to(device)

    # input batchsize*numsteps
    def forward(self, inputs, state):
        inputs = inputs.T.to(dtype=torch.int64)
        x = F.one_hot(inputs, self.vocab_size).type(torch.float32)
        Y, state = self.lstm(x, state)

        # 这个地方需要reshape 因为最后的y是1维的 而这个result 是三维要变成2维的 才可以满足交叉熵损失函数的要求
        result = self.linear(Y.reshape(-1,Y.shape[-1]))
        return result, state
    # 初始化隐状态
    def begin_state(self, batch_size, device):
        return torch.zeros((1,batch_size, hidden_size), device=device), torch.zeros((1,batch_size, hidden_size),
                                                                              device=device)

if __name__ == '__main__':
    batch_size = 32
    num_steps = 35
    hidden_size = 256
    vocab_size = 28
    num_epoch, lr = 500, 1
    date_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=False)
    device = 'cpu'
    ##net = RNNModelScratch(vocab_size, hidden_size, device, get_lstm_params, init_lstm_state, lstm)
    ##state = net.begin_state(batch_size, device)
    net=RNN(vocab_size, hidden_size, device)

    train_ch8(net, date_iter, vocab, lr, num_epoch, device)
