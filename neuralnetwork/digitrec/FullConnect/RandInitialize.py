import numpy as np


def initialise(a, b):
    epsilon = 0.15
    c = np.random.rand(a, b + 1) * (
        # 随机初始化权重在 [-epsilon, +epsilon] 范围内
            2 * epsilon) - epsilon
    return c
