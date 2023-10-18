import random

import numpy as np

from Model import neural_network
from Accuracy import accuracy
from Method import plot_loss_and_accuracy


def BGD(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    # 创建空列表来存储损失和准确度
    loss_history = []
    accuracy_history = []
    for i in range(iter_num):
        cost, grad = neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X,
                                    y, lambda_reg)
        nn_params -= alpha_rate * grad
        if (i % 10) == 0:
            print(f"Iteration {i}: Cost {cost}")
            print('Training Set Accuracy: {:f}'.format(
                accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)))
            loss_history.append(cost)
            accuracy_history.append(accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y))
    plot_loss_and_accuracy(loss_history, accuracy_history)
    return nn_params


def SGD(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    batch_size = 1
    m = X.shape[0]
    # 创建空列表来存储损失和准确度
    loss_history = []
    accuracy_history = []
    for i in range(iter_num):
        indices = list(range(m))
        random.shuffle(indices)
        totalcost = 0
        for j in range(0, m, batch_size):
            batch_indices = indices[j:j + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            cost, grad = neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X_batch, y_batch,
                                        lambda_reg)
            nn_params -= alpha_rate * grad
            totalcost += cost
        print(f"Iteration {i}: Cost {totalcost / m}")
        print('Training Set Accuracy: {:f}'.format(
            accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)))
        loss_history.append(totalcost / m)
        accuracy_history.append(accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y))
    plot_loss_and_accuracy(loss_history, accuracy_history)

    return nn_params


def OGD(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    batch_size = 32
    m = X.shape[0]
    # 创建空列表来存储损失和准确度
    loss_history = []
    accuracy_history = []
    for i in range(iter_num):
        indices = list(range(m))
        totalcost = 0
        for j in range(0, m, batch_size):
            batch_indices = indices[j:j + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            cost, grad = neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X_batch, y_batch,
                                        lambda_reg)
            totalcost += cost
            nn_params -= alpha_rate * grad
        print(f"Iteration {i}: Cost {totalcost / m}")
        print('Training Set Accuracy: {:f}'.format(
            accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)))
        loss_history.append(totalcost / m)
        accuracy_history.append(accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y))
    plot_loss_and_accuracy(loss_history, accuracy_history)

    return nn_params


def MiniBGD(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    batch_size = 64
    m = X.shape[0]
    # 创建空列表来存储损失和准确度
    loss_history = []
    accuracy_history = []
    for i in range(iter_num):
        # 随机打乱数据和标签，以创建随机的小批次
        indices = list(range(m))
        random.shuffle(indices)

        for j in range(0, m, batch_size):
            batch_indices = indices[j:j + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            cost, grad = neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X_batch, y_batch,
                                        lambda_reg)
            nn_params -= alpha_rate * grad

        if (i % 10) == 0:
            print(f"Iteration {i}: Cost {cost}")
            print('Training Set Accuracy: {:f}'.format(
                accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)))
            loss_history.append(cost)
            accuracy_history.append(accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y))
    plot_loss_and_accuracy(loss_history, accuracy_history)

    return nn_params


def Momentum(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    beta = 0.9
    # 初始化动量向量
    v = np.zeros(nn_params.shape)
    # 创建空列表来存储损失和准确度
    loss_history = []
    accuracy_history = []
    for i in range(iter_num):
        cost, grad = neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

        # 更新动量
        v = beta * v + alpha_rate * grad

        # 更新参数
        nn_params -= v

        if (i % 10) == 0:
            print(f"Iteration {i}: Cost {cost}")
            print('Training Set Accuracy: {:f}'.format(
                accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)))
            loss_history.append(cost)
            accuracy_history.append(accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y))
    plot_loss_and_accuracy(loss_history, accuracy_history)
    return nn_params


def Adagrad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    epsilon = 1e-8
    # 初始化梯度平方累积
    G = np.zeros(nn_params.shape)
    # 创建空列表来存储损失和准确度
    loss_history = []
    accuracy_history = []
    for i in range(iter_num):
        cost, grad = neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

        # 更新梯度平方累积
        G += np.square(grad)

        # 计算适应的学习率
        alpha = alpha_rate / np.sqrt(G + epsilon)

        # 更新参数
        nn_params -= alpha * grad

        if (i % 10) == 0:
            print(f"Iteration {i}: Cost {cost}")
            print('Training Set Accuracy: {:f}'.format(
                accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)))
            # 记录损失和准确度
            loss_history.append(cost)
            accuracy_history.append(accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y))
    plot_loss_and_accuracy(loss_history, accuracy_history)
    return nn_params


def Adam(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    # 初始化一阶矩和二阶矩
    m = np.zeros(nn_params.shape)
    v = np.zeros(nn_params.shape)
    t = 0
    # 创建空列表来存储损失和准确度
    loss_history = []
    accuracy_history = []

    for i in range(iter_num):
        cost, grad = neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

        t += 1
        # 更新一阶矩和二阶矩
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # 偏差修正
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # 计算适应的学习率
        alpha = alpha_rate / (np.sqrt(v_hat) + epsilon)

        # 更新参数
        nn_params -= alpha * m_hat

        if (i % 10) == 0:
            print(f"Iteration {i}: Cost {cost}")
            print('Training Set Accuracy: {:f}'.format(
                accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)))
            # 记录损失和准确度
            loss_history.append(cost)
            accuracy_history.append(accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y))
    plot_loss_and_accuracy(loss_history, accuracy_history)
    return nn_params


def Adamax(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    # 初始化一阶矩和 max 累积
    m = np.zeros(nn_params.shape)
    u = np.zeros(nn_params.shape)

    # 创建空列表来存储损失和准确度
    loss_history = []
    accuracy_history = []

    for i in range(iter_num):
        cost, grad = neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

        # 更新一阶矩和 max 累积
        m = beta1 * m + (1 - beta1) * grad
        u = np.maximum(beta2 * u, np.abs(grad))

        # 计算适应的学习率
        alpha = alpha_rate / (u + epsilon)

        # 更新参数
        nn_params -= alpha * m

        if (i % 10) == 0:
            print(f"Iteration {i}: Cost {cost}")
            print('Training Set Accuracy: {:f}'.format(
                accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)))
            # 记录损失和准确度
            loss_history.append(cost)
            accuracy_history.append(accuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y))
    plot_loss_and_accuracy(loss_history, accuracy_history)

    return nn_params
