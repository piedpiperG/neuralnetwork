import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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


class Run:

    def __init__(self, train_size, test_size, learning_rate=0.01, dropout_rate=0.3, epoch_num=2, batch_size=3):
        self.train_size = train_size
        self.test_size = test_size
        self.learning_rate = learning_rate  # 学习率
        self.dropout_rate = dropout_rate  # drop_out概率
        self.epoch_num = epoch_num  # 迭代次数
        self.batch_size = batch_size  # 批量大小
        self.data = Load_Data(train_size=self.train_size, test_size=self.test_size)

    def train(self):
        data = self.data
        X_train, y_train = data.load_train()

        learning_rate = self.learning_rate
        dropout_rate = self.dropout_rate
        epoch_num = self.epoch_num
        batch_size = self.batch_size

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
            for i in range(0, self.train_size, batch_size):
                X = X_train[i:i + batch_size]
                Y = y_train[i:i + batch_size]

                predict = conv1.forward(X)
                predict = batchnorm1.forward(predict)
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
                delta = batchnorm1.backward(delta, learning_rate)
                conv1.backward(delta, learning_rate)

                print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

            learning_rate *= 0.95 ** (epoch + 1)
            np.savez("data2.npz", k1=conv1.k, b1=conv1.b, k2=conv2.k, b2=conv2.b, w3=link1.W, b3=link1.b,
                     g1=batchnorm1.gamma, beta1=batchnorm1.beta, run_mean1=batchnorm1.running_mean,
                     run_var1=batchnorm1.running_var)

    def eval(self):
        r = np.load("data2.npz")
        data = self.data
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
        batchnorm1.gamma = r["g1"]
        batchnorm1.beta = r["beta1"]
        batchnorm1.running_mean = r["run_mean1"]
        batchnorm1.running_var = r["run_var1"]

        num = 0
        size = len(y_test)
        print(size)
        for i in range(size):
            X = X_test[i]
            X = X[np.newaxis, :]
            Y = y_test[i]

            predict = conv1.forward(X)
            predict = batchnorm1.forward(predict, training=False)
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
                print(f'{i + 1}:{num / (i + 1) * 100}')

        print("TEST-ACC: ", num / size * 100, "%")
        return num / size * 100


if __name__ == '__main__':
    # 定义超参数的不同组合
    learning_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0.2, 0.3, 0.4]
    epoch_nums = [2, 5]
    batch_sizes = [1, 3, 10, 20]

    # 为了绘制三维图表，我们需要把数据整理成三维坐标形式
    # 初始化坐标列表
    lr_list = []
    dr_list = []
    bs_list = []
    best_acc_list = []

    # 存储结果的字典
    results = {}
    # 进行交叉验证
    for lr in learning_rates:
        for dr in dropout_rates:
            for en in epoch_nums:
                for bs in batch_sizes:
                    # 初始化Run类的实例
                    run = Run(train_size=600, test_size=1000, learning_rate=lr, dropout_rate=dr, epoch_num=en,
                              batch_size=bs)
                    # 训练模型
                    run.train()
                    # 评估模型
                    accuracy = run.eval()
                    # 存储结果
                    results[(lr, dr, en, bs)] = accuracy
                    print(f"LR: {lr}, DR: {dr}, EN: {en}, BS: {bs}, Acc: {accuracy}")

    # 选取每组learning_rate, dropout_rate和batch_size组合下，epoch_num最好的结果
    for lr in learning_rates:
        for dr in dropout_rates:
            for bs in batch_sizes:
                # 找出当前组合下，epoch_num的最大准确率
                best_acc = max(results[(lr, dr, en, bs)] for en in epoch_nums)
                # 将当前组合和对应的最佳准确率添加到列表
                lr_list.append(lr)
                dr_list.append(dr)
                bs_list.append(bs)
                best_acc_list.append(best_acc)
    # 绘制三维图表
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    sc = ax.scatter(lr_list, dr_list, bs_list, c=best_acc_list, cmap='viridis')

    # 设置坐标轴
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Dropout Rate')
    ax.set_zlabel('Batch Size')

    # 添加颜色条表示准确率
    plt.colorbar(sc, label='Accuracy (%)')

    # 展示图表
    plt.show()
    # 创建数据字典
    data = {
        'Learning Rate': lr_list,
        'Dropout Rate': dr_list,
        'Batch Size': bs_list,
        'Accuracy': best_acc_list
    }

    # 创建DataFrame
    df = pd.DataFrame(data)
    # 使用DataFrame的sort_values方法按Accuracy列降序排序
    df_sorted = df.sort_values(by='Accuracy', ascending=False)
    df = df_sorted
    df.to_csv("results.csv")
    # 绘制表格，但不包含索引
    fig, ax = plt.subplots(figsize=(10, 3))  # 设置图表大小
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colLoc='center')

    # 调整表格大小
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.show()
