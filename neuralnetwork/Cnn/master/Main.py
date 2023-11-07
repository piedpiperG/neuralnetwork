import numpy as np
from utils import Load_Data
from model import Conv, Pool, Linear
from method import Dropout, BatchNorm


#  激活函数(使用ReLu)
class Relu:
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    # 作截断操作
    def backward(self, delta):
        delta[self.x < 0] = 0
        return delta


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

    def backward(self, delta, training=True):
        if training:
            return delta * self.mask
        else:
            return delta


def train():
    data = Load_Data()
    X_train, y_train = data.load_train()

    learning_rate = 0.01  # 学习率
    batch_size = 3  # 批量大小
    dropout_rate = 0.3  # drop_out概率
    epoch_num = 2  # 迭代次数

    conv1 = Conv(kernel_shape=(5, 5, 1, 6))
    relu1 = Relu()
    pool1 = Pool()
    conv2 = Conv(kernel_shape=(5, 5, 6, 16))  # 8x8x16
    relu2 = Relu()
    pool2 = Pool()  # 4x4x16
    link1 = Linear(256, 10)
    softmax = Softmax()
    dropout1 = Dropout(rate=dropout_rate)
    batchnorm1 = BatchNorm(6)

    for epoch in range(epoch_num):
        for i in range(0, 6000, batch_size):
            X = X_train[i:i + batch_size]
            Y = y_train[i:i + batch_size]

            predict = conv1.forward(X)
            # predict = batchnorm1.forward(predict)
            predict = relu1.forward(predict)
            predict = dropout1.forward(predict, training=True)
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
            delta = dropout1.backward(delta)
            delta = relu1.backward(delta)
            # delta = batchnorm1.backward(delta, learning_rate)
            conv1.backward(delta, learning_rate)

            print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

        learning_rate *= 0.95 ** (epoch + 1)
        np.savez("data2.npz", k1=conv1.k, b1=conv1.b, k2=conv2.k, b2=conv2.b, w3=link1.W, b3=link1.b)


def eval():
    r = np.load("data2.npz")
    data = Load_Data()
    X_test, y_test = data.load_test()

    conv1 = Conv(kernel_shape=(5, 5, 1, 6))  # 24x24x6
    batchnorm1 = BatchNorm(6)
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
        # predict = batchnorm1.forward(predict, training=False)
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


# 寻找最优超参数

# def cross_validate(X_train, y_train, param_grid, k=5):
#     best_params = None
#     best_score = float('inf')
#
#     for params in param_grid:
#         scores = []
#         for train_indices, val_indices in k_fold_split(k, len(X_train)):
#             X_train_kf, X_val_kf = X_train[train_indices], X_train[val_indices]
#             y_train_kf, y_val_kf = y_train[train_indices], y_train[val_indices]
#
#             # 构建并训练模型，你需要将此部分替换为实际模型训练和评估代码
#             model = build_model(params)
#             train_model(model, X_train_kf, y_train_kf, params)
#             score = evaluate_model(model, X_val_kf, y_val_kf)
#
#             scores.append(score)
#
#         avg_score = np.mean(scores)
#         if avg_score < best_score:
#             best_score = avg_score
#             best_params = params
#
#     return best_params

def Find_best():
    data = Load_Data()
    X_train, y_train = data.load_train()
    # 定义超参数网格
    param_grid = [
        {'learning_rate': 0.01, 'batch_size': 32, 'dropout_rate': 0.3},
        {'learning_rate': 0.001, 'batch_size': 64, 'dropout_rate': 0.5},

        # ... 在这里添加更多的超参数组合
    ]
    # 使用交叉验证搜索最佳超参数
    # best_params = cross_validate(X_train, y_train, param_grid)
    # print("Best hyperparameters found:", best_params)


if __name__ == '__main__':
    train()
    eval()
