import numpy as np
from scipy.io import loadmat


# 转化为onehot向量
def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][int(targets[i])] = 1
    return result


# 将卷积核展平展开,进行卷积运算
def img2col(x, ksize, stride):
    wx, hx, cx = x.shape  # [width,height,channel]
    feature_w = (wx - ksize) // stride + 1  # 返回的特征图尺寸
    image_col = np.zeros((feature_w * feature_w, ksize * ksize * cx))
    num = 0
    for i in range(feature_w):
        for j in range(feature_w):
            image_col[num] = x[i * stride:i * stride + ksize, j * stride:j * stride + ksize, :].reshape(-1)
            num += 1
    return image_col


class Load_Data:
    def __init__(self, train_size=60000, test_size=10000):
        '''
        导入数据集，划分为 60,000 个训练样本，10,000 个测试样本
        '''
        # 加载数据文件
        data = loadmat('mnist-original.mat')
        # 提取数据的特征矩阵，并进行转置
        X = data['data']
        self.X = X.transpose()
        # 然后将特征除以 255，重新缩放到 [0,1] 的范围内，以避免在计算过程中溢出
        # 从数据中提取 labels
        y = data['label']
        self.y = y.flatten()
        # 将数据分割为 60,000 个训练集
        self.train_size = train_size
        # 和 10,000 个测试集
        self.test_size = test_size
        self.indices = np.arange(70000)
        np.random.shuffle(self.indices)
        self.X = self.X[self.indices]
        self.y = self.y[self.indices]

    def load_train(self):
        X_train = self.X[:self.train_size, :].reshape(self.train_size, 28, 28)
        y_train = self.y[:self.train_size]
        X_train = X_train.reshape(self.train_size, 28, 28, 1) / 255.  # 输入向量处理
        y_train = onehot(y_train, self.train_size)  # 标签one-hot处理 (60000, 10)
        return X_train, y_train

    def load_test(self):
        X_test = self.X[self.train_size:self.train_size + self.test_size, :].reshape(self.test_size, 28, 28)
        y_test = self.y[self.train_size:self.train_size + self.test_size]
        X_test = X_test.reshape(self.test_size, 28, 28, 1) / 255.
        return X_test, y_test
