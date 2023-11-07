import numpy as np


class Dropout:
    def __init__(self, rate):
        self.rate = rate  # dropout的概率

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * self.mask
        else:
            return x

    def backward(self, delta, training=True):
        if training:
            return delta * self.mask
        else:
            return delta


class BatchNorm:
    def __init__(self, num_features):
        # Gamma: 缩放参数，Beta: 平移参数，都需要学习
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        # 运行时的均值和方差
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)
        # 用于反向传播的保存参数
        self.x_norm = None
        self.mu = None
        self.var = None
        self.eps = 1e-5  # 防止除以0

    def forward(self, x, training=True, momentum=0.9):
        if training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.x_norm = (x - mu) / np.sqrt(var + self.eps)
            out = self.gamma * self.x_norm + self.beta

            # 更新运行时均值和方差
            self.running_mean = momentum * self.running_mean + (1 - momentum) * mu
            self.running_var = momentum * self.running_var + (1 - momentum) * var

            # 保存反向传播需要的值
            self.mu = mu
            self.var = var
        else:
            # 测试时使用运行时的均值和方差
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, delta):
        N, H, W, C = delta.shape
        learning_rate = 0.3

        # 对gamma的梯度
        dgamma = np.sum(delta * self.x_norm, axis=(0, 2, 3)).reshape(self.gamma.shape)
        # 对beta的梯度
        dbeta = np.sum(delta, axis=(0, 2, 3)).reshape(self.beta.shape)

        # 对输入x的梯度
        dx_norm = delta * self.gamma.reshape(1, -1, 1, 1)
        dvar = np.sum(dx_norm * (self.x_norm - self.mu), axis=(0, 2, 3)) * -0.5 * np.power(
            self.var.reshape(-1) + self.eps, -1.5)
        dmu = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=(0, 2, 3)) + dvar * np.mean(
            -2 * (self.x_norm - self.mu), axis=(0, 2, 3))

        dx = (dx_norm / np.sqrt(self.var + self.eps)) + (
                dvar.reshape(1, C, 1, 1) * 2 * (self.x_norm - self.mu) / (N * H * W)) + (
                     dmu.reshape(1, C, 1, 1) / (N * H * W))

        # 更新gamma和beta
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dx


# Layer Normalization layer
class LayerNorm:
    def __init__(self, num_features):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(axis=1, keepdims=True)
        variance = x.var(axis=1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        out = self.gamma * x_normalized + self.beta
        self.cache = (x, x_normalized, mean, variance)
        return out

    def backward(self, dout):
        x, x_normalized, mean, variance = self.cache
        N = x.shape[1]

        dbeta = dout.sum(axis=0)
        dgamma = np.sum(dout * x_normalized, axis=0)

        dx_normalized = dout * self.gamma
        dvariance = np.sum(dx_normalized * (x - mean) * -0.5 * np.power(variance + self.eps, -1.5), axis=1,
                           keepdims=True)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(variance + self.eps), axis=1, keepdims=True) + \
                dvariance * np.mean(-2 * (x - mean), axis=1, keepdims=True)

        dx = (dx_normalized / np.sqrt(variance + self.eps)) + \
             (dvariance * 2 * (x - mean) / N) + \
             (dmean / N)

        return dx, dgamma, dbeta
