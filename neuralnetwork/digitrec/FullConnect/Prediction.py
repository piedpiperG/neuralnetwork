import numpy as np


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1)  # 给第一层加入偏置参数
    z2 = np.dot(X, Theta1.transpose())
    a2 = 1 / (1 + np.exp(-z2))  # 使用Sigmoid函数激活第二层
    one_matrix = np.ones((m, 1))
    a2 = np.append(one_matrix, a2, axis=1)  # 给第二层加入偏置参数
    z3 = np.dot(a2, Theta2.transpose())
    a3 = 1 / (1 + np.exp(-z3))  # 激活第三层
    p = (np.argmax(a3, axis=1))  # 输出预测的分类
    return p
