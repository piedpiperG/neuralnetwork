import numpy as np
from utils import img2col
from method import BatchNorm, LayerNorm


# 卷积层
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
        # self.batchnorm = BatchNorm(out_channel)

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
        # bk_img = self.batchnorm.forward(bk_img)
        return bk_img

    def backward(self, delta, learning_rate):
        # 批量，宽度，高度，输入通道数
        batch_size, in_width, in_height, in_channel = self.x.shape
        # 卷积核宽度，高度，卷积核输入通道数,卷积核个数
        kernel_width, kernel_height, kin_channel, kernel_num = self.k.shape
        # 从上一层传回来的梯度,批量,宽度,高度,卷积核个数
        delta_batch, delta_width, delta_height, delta_channel = delta.shape

        # delta = self.batchnorm.backward(delta)
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


# 全连接层
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
