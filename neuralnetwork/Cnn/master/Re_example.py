import numpy as np
import torchvision
from scipy.io import loadmat


def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][int(targets[i])] = 1
    return result


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


class Conv(object):
    # 初始化这一层网络中的参数
    def __init__(self, kernel_shape, stride=1):
        width, height, in_channel, out_channel = kernel_shape
        self.stride = stride  # 步长
        scale = np.sqrt(3 * in_channel * width * height / out_channel)  # 使用Xavier初始化缩放因子
        self.k = np.random.standard_normal(kernel_shape) / scale  # 初始化k，为一个四维向量
        self.b = np.random.standard_normal(out_channel) / scale  # 初始化b，为一个一维向量
        self.k_gradient = np.zeros(kernel_shape)  # 相应存储k的梯度
        self.b_gradient = np.zeros(out_channel)  # 相应存储b的梯度

    # 进行向前传播
    def forward(self, x):
        self.x = x
        # 批量，宽度，高度，输入通道数
        bx, wx, hx, cx = self.x.shape
        # 卷积核宽度，高度，输入数据通道数，个数
        wk, hk, ck, nk = self.k.shape  # kernel的宽、高、通道数、个数
        feature_w = (wx - wk) // self.stride + 1  # 返回的特征图尺寸
        # 返回图像的参数，批量，高，宽，通道
        feature = np.zeros((bx, feature_w, feature_w, nk))

        # 进行卷积的运算
        self.image_col = []
        kernel = self.k.reshape(-1, nk)
        for i in range(bx):
            image_col = img2col(self.x[i], wk, self.stride)
            feature[i] = (np.dot(image_col, kernel) + self.b).reshape(feature_w, feature_w, nk)
            self.image_col.append(image_col)
        return feature

    # 进行反向传播
    def backward(self, delta, learning_rate):
        bx, wx, hx, cx = self.x.shape  # batch,14,14,inchannel
        wk, hk, ck, nk = self.k.shape  # 5,5,inChannel,outChannel
        bd, wd, hd, cd = delta.shape  # batch,10,10,outChannel

        # 计算self.k_gradient,self.b_gradient
        delta_col = delta.reshape(bd, -1, cd)
        for i in range(bx):
            self.k_gradient += np.dot(self.image_col[i].T, delta_col[i]).reshape(self.k.shape)
        self.k_gradient /= bx
        self.b_gradient += np.sum(delta_col, axis=(0, 1))
        self.b_gradient /= bx

        # 计算delta_backward
        delta_backward = np.zeros(self.x.shape)
        k_180 = np.rot90(self.k, 2, (0, 1))  # numpy矩阵旋转180度
        k_180 = k_180.swapaxes(2, 3)
        k_180_col = k_180.reshape(-1, ck)

        if hd - hk + 1 != hx:
            pad = (hx - hd + hk - 1) // 2
            pad_delta = np.pad(delta, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        else:
            pad_delta = delta

        for i in range(bx):
            pad_delta_col = img2col(pad_delta[i], wk, self.stride)
            delta_backward[i] = np.dot(pad_delta_col, k_180_col).reshape(wx, hx, ck)

        # 反向传播
        self.k -= self.k_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward


# 池化层
class Pool(object):
    def forward(self, x):
        b, w, h, c = x.shape
        feature_w = w // 2
        feature = np.zeros((b, feature_w, feature_w, c))
        self.feature_mask = np.zeros((b, w, h, c))  # 记录最大池化时最大值的位置信息用于反向传播
        for bi in range(b):
            for ci in range(c):
                for i in range(feature_w):
                    for j in range(feature_w):
                        feature[bi, i, j, ci] = np.max(x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        index = np.argmax(x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        self.feature_mask[bi, i * 2 + index // 2, j * 2 + index % 2, ci] = 1
        return feature

    def backward(self, delta):
        return np.repeat(np.repeat(delta, 2, axis=1), 2, axis=2) * self.feature_mask


def train():
    # Mnist手写数字集
    '''
    导入数据集，划分为 60,000 个训练样本，10,000 个测试样本
    '''
    # 加载数据文件
    data = loadmat('mnist-original.mat')
    # 提取数据的特征矩阵，并进行转置
    X = data['data']
    X = X.transpose()
    # 然后将特征除以 255，重新缩放到 [0,1] 的范围内，以避免在计算过程中溢出
    # 从数据中提取 labels
    y = data['label']
    y = y.flatten()
    # 将数据分割为 60,000 个训练集
    train_size = 60000
    X_train = X[:train_size, :].reshape(train_size, 28, 28)
    y_train = y[:train_size]
    X_train = X_train.reshape(60000, 28, 28, 1) / 255.  # 输入向量处理
    y_train = onehot(y_train, 60000)  # 标签one-hot处理 (60000, 10)



if __name__ == '__main__':
    train()
