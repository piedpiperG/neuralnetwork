import numpy as np
from scipy.io import loadmat


class DataPrecessing(object):
    def loadFile(self):
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

        y_train = y_train[::-1].reshape(1, -1)
        y_test = y_test[::-1].reshape(1, -1)
        y_train = y_train[::-1].reshape(1, -1).astype(int)
        y_test = y_test[::-1].reshape(1, -1).astype(int)

        return x_train, y_train, x_test, y_test

    def Calculate_accuracy(self, target, prediction):
        score = 0
        for i in range(len(target)):
            if target[i] == prediction[i]:
                score += 1
        return score / len(target)

    def predict(self, test, weights):
        h = test * weights
        return h.argmax(axis=1)


def gradientAscent(feature_data, label_data, k, maxCycle, alpha):
    """Softmax分类器
    input:feature_data 手写数字像素矩阵
    label_data 对应的类别矩阵
    k 有多少个类别（10个）
    maxCycle 最大迭代次数
    alpha 学习率
    """
    Dataprecessing = DataPrecessing()
    x_train, y_train, x_test, y_test = Dataprecessing.loadFile()
    y_train = y_train.tolist()[0]
    y_test = y_test.tolist()[0]
    m, n = np.shape(feature_data)
    weights = np.mat(np.ones((n, k)))
    i = 0
    while i <= maxCycle:
        err = np.exp(feature_data * weights)
        if i % 100 == 0:
            print('cost score : ', cost(err, label_data))
            train_predict = Dataprecessing.predict(x_train, weights)
            test_predict = Dataprecessing.predict(x_test, weights)
            print('Train_accuracy : ', Dataprecessing.Calculate_accuracy(y_train, train_predict))
            print('Test_accuracy : ', Dataprecessing.Calculate_accuracy(y_test, test_predict))
        rowsum = -err.sum(axis=1)
        rowsum = rowsum.repeat(k, axis=1)
        err = err / rowsum
        for x in range(m):
            err[x, label_data[x]] += 1
        weights = weights + (alpha / m) * feature_data.T * err
        i += 1
    return weights


def cost(err, label_data):
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, label_data[i]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m


Dataprecessing = DataPrecessing()
x_train, y_train, x_test, y_test = Dataprecessing.loadFile()
y_train = y_train.tolist()[0]
gradientAscent(x_train, y_train, 10, 100000, 0.001)
