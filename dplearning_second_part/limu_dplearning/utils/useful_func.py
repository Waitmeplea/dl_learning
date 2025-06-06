import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display
from torch import nn


def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Timer(object):
    """
    记录多次运行的时间
    """

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并把时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear'
                 , yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1
                 , figsize=(3.2, 2.5)):
        if legend is None:
            legend = []
        ##subplot会返回fig和ax对象
        # Figure 是画布：想象你要创作一幅画，首先需要一块画布（Figure），它决定了画的整体大小和背景。
        # Axes 是画纸：在画布上，你可以贴多张画纸（Axes），每张画纸上绘制不同的内容（如折线图、柱状图）。
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]

        self.X, self.Y, self.fmts = None, None, fmts
        ##Lambda函数要求只能有一个表达式，而此处通过元组将多个方法调用合并为一个表达式，符合语法规则。
        self.config_axes = lambda: (
            self.axes[0].set_xlabel(xlabel),  # 设置X轴标签
            self.axes[0].set_ylabel(ylabel),  # 设置Y轴标签
            self.axes[0].set_xlim(*xlim) if xlim else None,  # 设置X轴范围（如 xlim=(0, 10)）
            self.axes[0].set_ylim(*ylim) if ylim else None,  # 设置Y轴范围
            self.axes[0].set_xscale(xscale),  # 设置X轴刻度类型（如 'linear', 'log'）
            self.axes[0].set_yscale(yscale),  # 设置Y轴刻度类型
            self.axes[0].legend(legend) if legend else None  # 显示图例（legend为图例参数，如 loc='best'）
        )

    def add(self, x, y):
        ###hasattr 动态检查一个对象是否包含指定属性或方法
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        self.config_axes()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        display.clear_output(wait=True)
        display.display(self.fig)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device(f'cpu')


def try_all_cpu():
    if torch.cuda.is_available():
        device = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        return device
    else:
        return [torch.device('cpu')]


class Accumulator:  #@save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(net.parameters()).device
    metric = Accumulator(2)
    for x, y in data_iter:
        if isinstance(x, list):
            x = [i.to(device) for i in x]
        else:
            x = x.to(device)
        y = y.to(device)
        metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device='cpu'):
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


import random
import collections
import re
import requests
import random
from torch.nn import functional as F

import math
import torch
from torch import nn
import sys

sys.path.append('../')

import torch
from torch import nn


##读取文章
def read_time_machine():
    with open(r'..\data\time_machine.txt', 'r') as f:
        content = f.readlines()
    return [re.sub('[^A-Za-z]', ' ', i.replace('\n', '')).strip().lower() for i in content]


## 定义一个拆分词元的函数 结果是词元组成的list
# 修改过了
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

##定义一个统计频率的函数 可以处理1d2d
def count_corpus(token_list):
    if isinstance(token_list[0], list):
        token_list = [token for i in token_list for token in i]
    tokens_count = collections.Counter(token_list)
    return tokens_count


class Vocal():
    def __init__(self, token_list=None, min_feq=0, reserved_tokens=None):
        self.token_list = token_list
        if token_list is None:
            self.token_list = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter_info = count_corpus(token_list)
        self._token_feq = []
        ##只接受符合条件的词
        for items in sorted(counter_info.items(), key=lambda x: x[1], reverse=True):
            if items[1] < min_feq:
                break
            else:
                self._token_feq.append(items)

        if '<unk>' in reserved_tokens:
            self.idx_to_token = reserved_tokens
        else:
            self.idx_to_token = ['unk'] + reserved_tokens

        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}
        for token, _ in self._token_feq:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.token_to_idx)

    def __len__(self):
        return len(self.token_to_idx)

    ##实现一个索引方法 但是传入的索引是token 返回是idx,保证未知token显示0值
    def __getitem__(self, tokens):
        if not isinstance(tokens, (tuple, list)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(i) for i in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):
        return 0

    @property
    def tokens_freq(self):
        return self._token_feq


def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    tokens = tokenize(read_time_machine(), 'char')
    vocal = Vocal(tokens)

    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocal[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocal


##产生随机序列
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    ##randint是左闭右闭 所以必须得减一不然0和numsteps实际上是重复的
    corpus = corpus[random.randint(0, num_steps - 1):]
    ##因为y要比x多一位 一共可以有这么多子序列
    num_subseqs = (len(corpus) - 1) // num_steps

    ## 然后找出每一个num seq的起始索引拿出来放到列表里 这里不能用corpus的长度 而是num_steps*num_subseqs
    seq_start_index = [i for i in range(0, num_steps * num_subseqs, num_steps)]
    ##打乱索引
    random.shuffle(seq_start_index)

    ## 再除以batchsize 有多少个batch
    num_batch = num_subseqs // batch_size
    ### 然后给每个batch找x和y 索引起始位置在seq_start_index里
    for i in range(num_batch):
        index_list = seq_start_index[i * batch_size:(i + 1) * batch_size]
        x = []
        y = []
        for _index in index_list:
            x.append(corpus[_index:_index + num_steps])
            y.append(corpus[_index + 1:_index + 1 + num_steps])
        yield torch.tensor(x), torch.tensor(y)


# def seq_data_iter_sequential(corpus,batch_size,num_steps):
#     random.seed(42)
#     corpus=corpus[random.randint(0,num_steps):]
#     batch_num=(len(corpus)-1)//(batch_size*num_steps)
#     Xs=torch.tensor(corpus[:batch_num*batch_size*num_steps]).reshape(-1,batch_size,num_steps)
#     Ys=torch.tensor(corpus[1:batch_num*batch_size*num_steps+1]).reshape(-1,batch_size,num_steps)
#     for i in range(batch_num):
#         yield Xs[i],Ys[i]
# s1=seq_data_iter_sequential(corpus,batch_size,num_steps)
# 这个方案是错的 连续要求在不同的batch上保持连续 而不是在一个batch的多个样本上保持连续 并且 batch_num=(len(corpus)-1)//(batch_size*num_steps) 这样也不太好
#因为batch_num 计算方式耦合了 batch_size 和 num_steps，


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    random.seed(0)
    corpus = corpus[random.randint(0, num_steps):]
    num_tokens = (len(corpus) - 1) // batch_size * batch_size  ## 保证是batch_size的倍数
    ###batch_size需要放在最外维度：常规样本维度安排 外层是样本个数 也就是batch_size
    Xs = torch.tensor(corpus[:num_tokens]).reshape(batch_size, -1)
    Ys = torch.tensor(corpus[1:num_tokens + 1]).reshape(batch_size, -1)
    ##维度1是每个batchsize有多少token
    batch_num = Xs.shape[1] // num_steps
    for i in range(batch_num):
        yield Xs[:, i * num_steps:(i + 1) * num_steps], Ys[:, i * num_steps:(i + 1) * num_steps]


class SeqDataLoader:  #@save
    def __init__(self, batch_size, num_steps, use_random_iter, max_token):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocal = load_corpus_time_machine(max_token)
        self.batch_size, self.num_steps = batch_size, num_steps

    ##__iter__方法使得整个类变成可迭代的
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_token=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_token)
    return data_iter, data_iter.vocal


def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    ##outputs只有第一个字母
    outputs = [vocab[prefix[0]]]
    ## 定义一个获取最新output的函数
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        ##get_input()每次只有一个值 作为输入 state为隐状态
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    norm = torch.sqrt(torch.sum(torch.tensor([torch.sum(p.grad ** 2) for p in params])))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_utils`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, Timer()
    metric = Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 初始化状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 非原地分离状态
            if isinstance(net, nn.Module):
                if isinstance(state, tuple):
                    # LSTM状态：分离每个张量
                    state = tuple(s.detach() for s in state)
                else:
                    # GRU状态：直接分离
                    state = state.detach()
            else:
                # 自定义模型状态处理
                state = (s.detach() for s in state)
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()

    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


###########

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    if X.dim() != 2 or valid_len.dim() != 1:
        raise ValueError('Expect 2d tensor')

    # 首先x是二维的 最内层维度是句子长度 注意：是训练集所以才知道句子真实长度
    # 拿出总的长度 得到长度

    maxlen = X.shape[1]
    # 然后用总长度生成一个1维的向量 使用函数扩展成2维以便与valid_len进行广播
    mask = torch.unsqueeze(torch.arange(0, maxlen, dtype=torch.long), dim=0)
    # mask在0维度扩充 valid在1维度扩充 因为每一个valid对应的是每一个x valid的数字其实是x的第二维向量
    mask = (mask < torch.unsqueeze(valid_len, dim=1))  # 这里小于号就够了 因为<eos>所在位置的索引其实是valid_len-1
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X为3D batch_size*x*y valid_len 最后一个轴的有效长度
    # 如果valid_len是1d 则len必须等于X最外围的长度
    # 如果valid_len是2d 则按批次计算有效长度 reshape后必须等于 x的0层 *1层长度
    if valid_lens is None:
        return torch.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 加性注意力
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super(AdditiveAttention,self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=False)
        self.W_v = nn.Linear(num_hiddens,1,bias=False)
        self.dropout = nn.Dropout(dropout)
    # queries 维度应该是 batch_size * 要查询的数量 * q_size向量长度
    # keys 维度是 batch_size * keys的数量（key-value)键值对 * key向量长度
    # values与key相等 value_size可以不一样
    def forward(self, queries, keys, values, valid_lens): ## valid_len从输入来的 屏蔽掉填充部分
        queries,keys=self.W_q(queries),self.W_k(keys)
        queries=queries.unsqueeze(2)
        keys=keys.unsqueeze(1)
        features = queries + keys
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        attention_temp=self.dropout(self.attention_weights)
        return torch.bmm(attention_temp, values)

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_out, *args, **kwargs):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_out = self.encoder(enc_X)
        init_state = self.decoder.init_state(enc_out)
        return self.decoder(dec_X, init_state)


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hidden, layers_num, dropout, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, num_hidden, layers_num, dropout=dropout)

    def forward(self, X, *args):
        embed_x = self.embedding(X)
        embed_x = embed_x.permute(1, 0, 2)
        output, hidden = self.gru(embed_x)
        return output, hidden





def read_data_nmt():
    with open(r'../data/fra-eng/fra.txt', 'r', encoding='utf-8') as f:
        result = f.read()
    return result



#### 英法语 序列预处理
# @save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    text = preprocess_nmt(read_data_nmt())
    src, tgt = [], []
    for i, lines in enumerate(text.split('\n')):
        if num_examples and i >= num_examples:
            break
        lines = lines.split('\t')
        if len(lines) == 2:
            src.append(lines[0].split(' '))
            tgt.append(lines[1].split(' '))
    return src, tgt


def truncate_pad(content, lens, pad_content):
    content = content[:lens]
    return content + [pad_content] * (lens - len(content))


## 将文本数据作为批量数据，并且加入<eos>至末尾 同时统计有效字符 包含eos
# 传入的二维list
def build_array_nmt(lines, vocab, num_steps):
    lines = [i + ['<eos>'] for i in lines]
    valid_lens = [len(i) for i in lines]
    lines = [truncate_pad(i, num_steps, '<pad>') for i in lines]
    lines = [vocab[i] for i in lines]
    return torch.tensor(lines), torch.tensor(valid_lens)


if __name__ == '__main__':

    timer = Timer()
    time.sleep(3)
    print(timer.stop())

    animator = Animator()
    for epoch in range(10):
        # 模拟两条曲线的数据（例如训练损失和验证损失）
        train_loss = 2 / (epoch + 1)
        val_loss = 3 / (epoch + 1)
        animator.add(x=epoch, y=[train_loss, val_loss])
