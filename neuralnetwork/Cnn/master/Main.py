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


# 加载数据集
class Load_Data:
    def __init__(self):
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
        self.train_size = 60000
        # 和 10,000 个测试集
        self.test_size = 10000
        # 生成随机排列的索引
        self.train_indices = np.arange(self.train_size)
        np.random.shuffle(self.train_indices)
        self.test_indices = np.arange(self.test_size)
        np.random.shuffle(self.test_indices)

    def load_train(self):
        X_train = self.X[:self.train_size, :].reshape(self.train_size, 28, 28)
        y_train = self.y[:self.train_size]
        X_train = X_train.reshape(60000, 28, 28, 1) / 255.  # 输入向量处理
        y_train = onehot(y_train, 60000)  # 标签one-hot处理 (60000, 10)
        # 使用随机排列的索引来打乱训练数据和标签
        X_train = X_train[self.train_indices]
        y_train = y_train[self.train_indices]
        return X_train, y_train

    def load_test(self):
        X_test = self.X[self.train_size:self.train_size + self.test_size, :].reshape(self.test_size, 28, 28)
        y_test = self.y[self.train_size:self.train_size + self.test_size]
        X_test = X_test.reshape(10000, 28, 28, 1) / 255.
        # 使用随机排列的索引来打乱训练数据和标签
        X_test = X_test[self.test_indices]
        y_test = y_test[self.test_indices]
        return X_test, y_test


# 卷积层类
class Conv:
    def __init__(self, kernel_shape, path=1):
        width, height, in_channel, out_channel = kernel_shape
        self.path = path
        scale = np.sqrt(3 * in_channel * width * height / out_channel)  # 使用Xavier初始化缩放因子
        self.k = np.random.standard_normal(kernel_shape) / scale  # 初始化k，为一个四维向量
        self.b = np.random.standard_normal(out_channel) / scale  # 初始化b，为一个一维向量
        self.k_gradient = np.zeros(kernel_shape)  # 相应存储k的梯度
        self.b_gradient = np.zeros(out_channel)  # 相应存储b的梯度
        pad = 0
        self.pad = pad

    # 进行前向传播
    def forward(self, x):
        self.x = x
        if self.pad != 0:
            self.x = np.pad(self.x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant')
        # 批量，宽度，高度，输入通道数
        batch_size, in_width, in_height, in_channel = self.x.shape
        # 卷积核宽度，高度，卷积核输入通道数,卷积核个数
        kernel_width, kernel_height, kin_channel, kernel_num = self.k.shape
        # 返回图像的参数，批量，高，宽，通道
        bkimg_size = (in_width - kernel_width) // self.path + 1
        bk_img = np.zeros((batch_size, bkimg_size, bkimg_size, kernel_num))

        # 进行卷积运算
        self.image_col = []
        kernel = self.k.reshape(-1, kernel_num)
        for i in range(batch_size):
            image_col = img2col(self.x[i], kernel_width, self.path)
            bk_img[i] = (np.dot(image_col, kernel) + self.b).reshape(bkimg_size, bkimg_size, kernel_num)
            self.image_col.append(image_col)
        return bk_img

    def backward(self, delta, learning_rate):
        # 批量，宽度，高度，输入通道数
        batch_size, in_width, in_height, in_channel = self.x.shape
        # 卷积核宽度，高度，卷积核输入通道数,卷积核个数
        kernel_width, kernel_height, kin_channel, kernel_num = self.k.shape
        # 从上一层传回来的梯度,批量,宽度,高度,卷积核个数
        delta_batch, delta_width, delta_height, delta_channel = delta.shape

        # 计算self.k_gradient,self.b_gradient
        delta_col = delta.reshape(delta_batch, -1, delta_channel)
        for i in range(batch_size):
            self.k_gradient += np.dot(self.image_col[i].T, delta_col[i]).reshape(self.k.shape)
        self.k_gradient /= batch_size
        self.b_gradient += np.sum(delta_col, axis=(0, 1))
        self.b_gradient /= batch_size

        # 计算delta_backward
        delta_backward = np.zeros(self.x.shape)
        k_180 = np.rot90(self.k, 2, (0, 1))  # numpy矩阵旋转180度
        k_180 = k_180.swapaxes(2, 3)
        k_180_col = k_180.reshape(-1, kin_channel)

        if delta_height - kernel_height + 1 != in_height:
            pad = (in_height - delta_height + kernel_height - 1) // 2
            pad_delta = np.pad(delta, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        else:
            pad_delta = delta

        for i in range(batch_size):
            pad_delta_col = img2col(pad_delta[i], kernel_width, self.path)
            delta_backward[i] = np.dot(pad_delta_col, k_180_col).reshape(in_width, in_height, in_channel)

        # 反向传播
        self.k -= self.k_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward


# 池化层
class Pool:
    # 向前传播
    def forward(self, x):
        batch_size, in_width, in_height, in_channel = x.shape
        bkimg_size = in_width // 2
        bk_img = np.zeros((batch_size, bkimg_size, bkimg_size, in_channel))
        # 记录池化位置,在反向传播中使用
        self.max_address = np.zeros((batch_size, in_width, in_height, in_channel))
        for b in range(batch_size):
            for c in range(in_channel):
                for w in range(bkimg_size):
                    for h in range(bkimg_size):
                        bk_img[b, w, h, c] = np.max(x[b, w * 2:w * 2 + 2, h * 2: h * 2 + 2, c])
                        index = np.argmax(x[b, w * 2:w * 2 + 2, h * 2: h * 2 + 2, c])
                        self.max_address[b, w * 2 + index // 2, h * 2 + index // 2, c] = 1
        return bk_img

    def backward(self, delta):
        return np.repeat(np.repeat(delta, 2, axis=1), 2, axis=2) * self.max_address

    #  激活函数(使用ReLu)


class Relu:
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    # 作截断操作
    def backward(self, delta):
        delta[self.x < 0] = 0
        return delta


class Linear(object):
    def __init__(self, inChannel, outChannel):
        scale = np.sqrt(inChannel / 2)
        self.W = np.random.standard_normal((inChannel, outChannel)) / scale
        self.b = np.random.standard_normal(outChannel) / scale
        self.W_gradient = np.zeros((inChannel, outChannel))
        self.b_gradient = np.zeros(outChannel)

    def forward(self, x):
        self.x = x
        x_forward = np.dot(self.x, self.W) + self.b
        return x_forward

    def backward(self, delta, learning_rate):
        ## 梯度计算
        batch_size = self.x.shape[0]
        self.W_gradient = np.dot(self.x.T, delta) / batch_size  # bxin bxout
        self.b_gradient = np.sum(delta, axis=0) / batch_size
        delta_backward = np.dot(delta, self.W.T)  # bxout inxout
        ## 反向传播
        self.W -= self.W_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward


# Softmax层输出分类
class Softmax(object):
    def cal_loss(self, predict, label):
        batchsize, classes = predict.shape
        self.predict(predict)
        loss = 0
        delta = np.zeros(predict.shape)
        for i in range(batchsize):
            delta[i] = self.softmax[i] - label[i]
            loss -= np.sum(np.log(self.softmax[i]) * label[i])
        loss /= batchsize
        return loss, delta

    def predict(self, predict):
        batchsize, classes = predict.shape
        self.softmax = np.zeros(predict.shape)
        for i in range(batchsize):
            predict_tmp = predict[i] - np.max(predict[i])
            predict_tmp = np.exp(predict_tmp)
            self.softmax[i] = predict_tmp / np.sum(predict_tmp)
        return self.softmax


def train():
    data = Load_Data()
    X_train, y_train = data.load_train()

    conv1 = Conv(kernel_shape=(5, 5, 1, 6))
    relu1 = Relu()
    pool1 = Pool()
    conv2 = Conv(kernel_shape=(5, 5, 6, 16))  # 8x8x16
    relu2 = Relu()
    pool2 = Pool()  # 4x4x16
    link1 = Linear(256, 10)
    softmax = Softmax()

    learning_rate = 0.01
    batch_size = 3
    for epoch in range(2):
        for i in range(0, 60000, batch_size):
            X = X_train[i:i + batch_size]
            Y = y_train[i:i + batch_size]

            predict = conv1.forward(X)
            predict = relu1.forward(predict)
            predict = pool1.forward(predict)
            predict = conv2.forward(predict)
            predict = relu2.forward(predict)
            predict = pool2.forward(predict)
            predict = predict.reshape(batch_size, -1)
            predict = link1.forward(predict)

            loss, delta = softmax.cal_loss(predict, Y)

            delta = link1.backward(delta, learning_rate)
            delta = delta.reshape(batch_size, 4, 4, 16)
            delta = pool2.backward(delta)
            delta = relu2.backward(delta)
            delta = conv2.backward(delta, learning_rate)
            delta = pool1.backward(delta)
            delta = relu1.backward(delta)
            conv1.backward(delta, learning_rate)

            print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

        learning_rate *= 0.95 ** (epoch + 1)
        np.savez("data2.npz", k1=conv1.k, b1=conv1.b, k2=conv2.k, b2=conv2.b, w3=link1.W, b3=link1.b)
    Theta = {
        "k1": conv1.k,
        "b1": conv1.b,
        "k2": conv2.k,
        "b2": conv2.b,
        "w3": link1.W,
        "b3": link1.b
    }
    return Theta


def eval():
    r = np.load("data2.npz")
    data = Load_Data()
    X_test, y_test = data.load_test()

    conv1 = Conv(kernel_shape=(5, 5, 1, 6))  # 24x24x6
    relu1 = Relu()
    pool1 = Pool()  # 12x12x6
    conv2 = Conv(kernel_shape=(5, 5, 6, 16))  # 8x8x16
    relu2 = Relu()
    pool2 = Pool()  # 4x4x16
    link1 = Linear(256, 10)
    softmax = Softmax()

    conv1.k = r["k1"]
    conv1.b = r["b1"]
    conv2.k = r["k2"]
    conv2.b = r["b2"]
    link1.W = r["w3"]
    link1.n = r["b3"]

    num = 0
    for i in range(10000):
        X = X_test[i]
        X = X[np.newaxis, :]
        Y = y_test[i]

        predict = conv1.forward(X)
        predict = relu1.forward(predict)
        predict = pool1.forward(predict)
        predict = conv2.forward(predict)
        predict = relu2.forward(predict)
        predict = pool2.forward(predict)
        predict = predict.reshape(1, -1)
        predict = link1.forward(predict)

        predict = softmax.predict(predict)

        if np.argmax(predict) == Y:
            num += 1
            print(i)

    print("TEST-ACC: ", num / 10000 * 100, "%")


if __name__ == '__main__':
    train()
    eval()
