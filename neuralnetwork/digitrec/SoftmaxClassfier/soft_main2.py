import numpy as np
from keras.datasets import mnist
from scipy.io import loadmat


def loadFile1():
    (x_train, x_target_tarin), (x_test, x_target_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
    x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
    x_train = np.mat(x_train)
    x_test = np.mat(x_test)
    x_target_tarin = np.mat(x_target_tarin)
    x_target_test = np.mat(x_target_test)
    print(x_target_tarin)
    print(x_target_test)

def loadFile():
    # 加载数据文件
    data = loadmat('mnist-original.mat')
    # 提取数据的特征矩阵，并进行转置，将二维数组转换为一维数组的形式
    x = data['data']
    x = x.transpose()
    # 然后将特征除以255.0，以浮点数的形式重新缩放到[0,1]的范围内，以避免在计算过程中溢出
    x = x / 255.0
    # 从数据中提取labels，即x数字像素对应的数字类别y
    y = data['label']
    y = y.flatten()
    # 将数据分割为60,000个训练集
    x_train = x[:60000, :]
    y_train = y[:60000]
    # 和10,000个测试集
    x_test = x[60000:, :]
    y_test = y[60000:]
    y_train = y_train[::-1].reshape(1, -1).astype(int)
    y_test = y_test[::-1].reshape(1, -1).astype(int)
    print(y_train)
    print(y_test)



loadFile1()
loadFile()