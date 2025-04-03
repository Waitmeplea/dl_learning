import math
import time
import numpy as np
import torch
from d2l import torch as d2l


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


if __name__ == '__main__':
    timer = Timer()
    time.sleep(3)
    print(timer.stop())
