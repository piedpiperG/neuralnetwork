from scipy.io import loadmat
import numpy as np

if __name__ == '__main__':
    '''
    导入数据集，划分为 60,000 个训练样本，10,000 个测试样本
    '''
    # 加载数据文件
    data = loadmat('mnist-original.mat')
    # 提取数据的特征矩阵，并进行转置
    X = data['data']
    X = X.transpose()
    # 然后将特征除以 255，重新缩放到 [0,1] 的范围内，以避免在计算过程中溢出
    X = X / 255
    # 从数据中提取 labels
    y = data['label']
    y = y.flatten()
    # 将数据分割为 60,000 个训练集
    train_size = 60000
    X_train = X[:train_size, :].reshape(train_size, 28, 28)
    y_train = y[:train_size]
    # 和 10,000 个测试集
    test_size = 10000
    X_test = X[train_size:train_size + test_size, :].reshape(test_size, 28, 28)
    y_test = y[train_size:train_size + test_size]
    print(X_train.shape)
    print(y_train.shape)
    print(y_train)
