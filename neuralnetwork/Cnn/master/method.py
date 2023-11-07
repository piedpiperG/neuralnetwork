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

    def backward(self, delta):
        return delta * self.mask


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
        self.x_reshaped = None

    def forward(self, x, training=True, momentum=0.9):
        N, H, W, C = x.shape
        # 确保gamma和beta的形状可以在通道维度上广播
        self.gamma = self.gamma.reshape(1, 1, 1, C)
        self.beta = self.beta.reshape(1, 1, 1, C)
        # 转换为(N*H*W, C)以计算每个通道的均值和方差
        x_reshaped = x.reshape((N * H * W, C))
        self.x_reshaped = x_reshaped

        if training:
            mu = x_reshaped.mean(axis=0)
            var = x_reshaped.var(axis=0)

            self.x_norm = (x_reshaped - mu) / np.sqrt(var + self.eps)
            self.x_norm = self.x_norm.reshape(N, H, W, C)

            self.running_mean = momentum * self.running_mean + (1 - momentum) * mu
            self.running_var = momentum * self.running_var + (1 - momentum) * var

            self.mu = mu
            self.var = var
        else:
            x_norm = (x_reshaped - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.x_norm = x_norm.reshape(N, H, W, C)
        out = self.gamma * self.x_norm + self.beta
        return out

    def backward(self, delta, learning_rate):
        N, H, W, C = delta.shape
        # 首先，转换delta的形状以匹配归一化数据的形状
        delta_reshaped = delta.reshape((N * H * W, C))
        x_norm_reshaped = self.x_norm.reshape((N * H * W, C))

        # 计算beta和gamma的梯度
        dbeta = delta_reshaped.sum(axis=0)
        dgamma = np.sum(x_norm_reshaped * delta_reshaped, axis=0)

        # 计算归一化数据的梯度
        dx_norm = delta_reshaped * self.gamma

        # 归一化x的梯度
        dvar = np.sum(dx_norm * (self.x_reshaped - self.mu), axis=0) * -0.5 * np.power(self.var + self.eps, -1.5)
        dmu = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=0) + dvar * np.mean(
            -2 * (self.x_reshaped - self.mu), axis=0)

        # 归一化梯度
        dx_reshaped = (dx_norm / np.sqrt(self.var + self.eps)) + (dvar * 2 * (self.x_reshaped - self.mu) / N) + (
                    dmu / N)
        dx = dx_reshaped.reshape(N, H, W, C)
        self.gamma = self.gamma - learning_rate * dgamma
        self.beta = self.beta - learning_rate * dbeta
        return dx



def k_fold_split(k, data_size):
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, data_size // k, dtype=int)
    fold_sizes[:data_size % k] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        mask = np.ones(data_size, dtype=bool)
        mask[indices[start:stop]] = False
        train_indices, val_indices = indices[mask], indices[start:stop]
        yield train_indices, val_indices
        current = stop
