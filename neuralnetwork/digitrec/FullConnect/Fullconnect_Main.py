import random

from scipy.io import loadmat
import numpy as np


# 初始化权重，b->a，给b层中加入一个偏置节点
def initialise(a, b):
    epsilon = 0.15
    c = np.random.rand(a, b + 1) * (
        # 随机初始化权重在 [-epsilon, +epsilon] 范围内
            2 * epsilon) - epsilon
    return c


def loss(Theta1, Theta2, y_vect, a3, lamb, m):
    # 计算损失值
    # 正则化方法选择
    # L2正则化
    L2_Reg = (lamb / (2 * m)) * (
            np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))
    # L1正则化
    L1_Reg = (lamb / (2 * m)) * (
            np.sum(np.abs(Theta1[:, 1:])) + np.sum(np.abs(Theta2[:, 1:]))
    )
    Reg = L2_Reg
    # 交叉熵损失（Cross-Entropy Loss）
    J = (1 / m) * (np.sum(np.sum(-y_vect * np.log(a3) - (1 - y_vect) * np.log(1 - a3)))) + Reg
    # 均方误差损失（Mean Squared Error Loss）
    # J = (1 / (2 * m)) * np.sum(np.sum(np.square(a3 - y_vect))) + Reg
    # # Hinge Loss（合页损失，用于支持向量机等）
    # J = (1 / m) * np.sum(np.maximum(0, 1 - (2 * y_vect - 1) * a3)) + Reg
    # #  Huber Loss（用于回归问题的平滑损失）
    # delta = 1.0  # 控制损失函数的平滑度
    # J = (1 / m) * np.sum(
    #     np.where(np.abs(a3 - y_vect) < delta, 0.5 * np.square(a3 - y_vect), delta * np.abs(a3 - y_vect))) + Reg
    return J


# 进行一次向前传播和向后传播，并返回损失和梯度
def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    # 分割获得三个层次之间两两的权重
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))

    # 向前传播
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1)  # 向输入层添加偏置单元，使之成为偏差节点
    a1 = X
    z2 = np.dot(X, Theta1.transpose())
    a2 = 1 / (1 + np.exp(-z2))  # 采用Sigmoid函数对隐藏层进行激活
    one_matrix = np.ones((m, 1))
    a2 = np.append(one_matrix, a2, axis=1)  # 向隐藏层添加偏置单元，使之成为偏差节点
    z3 = np.dot(a2, Theta2.transpose())
    a3 = 1 / (1 + np.exp(-z3))  # 采用Sigmoid函数对输出层进行激活

    # 将标签改为一个长度为10的布尔向量，在向量的10个布尔数值里，哪个数等于1，它就代表着几
    y_vect = np.zeros((m, 10))
    for i in range(m):
        y_vect[i, int(y[i])] = 1

    # 计算损失值
    J = loss(Theta1, Theta2, y_vect, a3, lamb, m)

    # 向后传播
    Delta3 = a3 - y_vect
    Delta2 = np.dot(Delta3, Theta2) * a2 * (1 - a2)
    Delta2 = Delta2[:, 1:]

    # 计算梯度
    Theta1[:, 0] = 0
    Theta1_grad = (1 / m) * np.dot(Delta2.transpose(), a1) + (lamb / m) * Theta1
    Theta2[:, 0] = 0
    Theta2_grad = (1 / m) * np.dot(Delta3.transpose(), a2) + (lamb / m) * Theta2
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return J, grad


# 进行一次正向传播来得到结果，用于预测准确率
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


def BGD(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    for i in range(iter_num):
        cost, grad = neural_network(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X,
                                    y, lambda_reg)
        nn_params -= alpha_rate * grad
        if (i % 10) == 0:
            print(f"Iteration {i}: Cost {cost}")
    return nn_params


def SGD(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    batch_size = 1
    m = X.shape[0]
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
        print(f"Iteration {i}: Cost {totalcost/m}")

    return nn_params


def OGD(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    batch_size = 1
    m = X.shape[0]
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
    print(f"Iteration {i}: Cost {totalcost/m}")


    return nn_params


def MiniBGD(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg, iter_num, alpha_rate):
    batch_size = 64
    m = X.shape[0]
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
    return nn_params


if __name__ == '__main__':
    '''
    导入数据集，划分为60，000个训练样本，10，000个测试样本
    '''
    # 加载数据文件
    data = loadmat('mnist-original.mat')
    # 提取数据的特征矩阵，并进行转置
    X = data['data']
    X = X.transpose()
    # 然后将特征除以255，重新缩放到[0,1]的范围内，以避免在计算过程中溢出
    X = X / 255
    # 从数据中提取labels
    y = data['label']
    y = y.flatten()
    # 将数据分割为60,000个训练集
    train_size = 60000
    X_train = X[:train_size, :]
    y_train = y[:train_size]
    # 和10,000个测试集
    test_size = 10000
    X_test = X[train_size:train_size + test_size, :]
    y_test = y[train_size:train_size + test_size]
    '''
    构建三层全连接神经网络的参数
    '''
    # 输入层，隐藏层，输出层节点个数
    input_layer_size = 784  # 图片大小为 (28 X 28) px 所以设置784个特征
    hidden_layer_size = 16
    num_labels = 10  # 拥有十个标准为 [0, 9] 十个数字
    # 初始化层之间的权重 Thetas
    initial_Theta1 = initialise(hidden_layer_size, input_layer_size)  # 输入层和隐藏层之间的权重
    initial_Theta2 = initialise(num_labels, hidden_layer_size)  # 隐藏层和输出层之间的权重
    # 设置神经网络的参数
    initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
    lambda_reg = 0.1  # 避免过拟合
    '''
    进行神经网络的训练
    '''
    # 设置学习率和迭代次数
    alpha = 0.1
    max_iter = 3
    # 训练神经网络，根据函数选择优化方法
    initial_nn_params = OGD(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train,
                                lambda_reg, max_iter, alpha)


    # 重新分割，获得三个层次之间两两的权重
    Theta1 = np.reshape(initial_nn_params[:hidden_layer_size * (input_layer_size + 1)], (
        hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
    Theta2 = np.reshape(initial_nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))  # shape = (10, 101)
    # 测试集的准确度
    pred = predict(Theta1, Theta2, X_test)
    print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100)))
    # 训练集的准确度
    pred = predict(Theta1, Theta2, X_train)
    print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))
