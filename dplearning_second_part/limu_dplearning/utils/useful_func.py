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


if __name__ == '__main__':
    train_ch6()
    timer = Timer()
    time.sleep(3)
    print(timer.stop())

    animator = Animator()
    for epoch in range(10):
        # 模拟两条曲线的数据（例如训练损失和验证损失）
        train_loss = 2 / (epoch + 1)
        val_loss = 3 / (epoch + 1)
        animator.add(x=epoch, y=[train_loss, val_loss])
