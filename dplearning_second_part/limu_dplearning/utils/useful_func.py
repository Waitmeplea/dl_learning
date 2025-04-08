import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display

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