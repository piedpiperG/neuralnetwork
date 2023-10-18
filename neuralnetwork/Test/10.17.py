import numpy as np
from scipy.io import loadmat

# 加载数据文件
data = loadmat('mnist-original.mat')

# 提取数据的特征矩阵，并进行转置
X = data['data']
X = X.transpose()

# 然后将特征除以255，重新缩放到[0,1]的范围内，以避免在计算过程中溢出,得到28*28的向量
X = X / 255

# 从数据中提取labels
y = data['label']
y = y.flatten()

# 将数据分割为60,000个训练集
X_train = X[:60000, :]
y_train = y[:60000]

# 和10,000个测试集
X_test = X[60000:, :]
y_test = y[60000:]


# 搭建神经网络
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes  # 输入层
        self.hnodes = hiddennodes  # 隐藏层
        self.onodes = outputnodes  # 输出层
        self.lr = learningrate  # 学习率
        # 设置参数w1,w2,b1,b2，并随机生成这些参数。
        self.w1 = np.random.randn(self.inodes, self.hnodes)
        self.b1 = np.zeros(self.hnodes)
        self.w2 = np.random.randn(self.hnodes, self.onodes)
        self.b2 = np.zeros(self.onodes)

    def Backpropagation(self, batch_images, batch_labels):
        # 前向传播
        z1 = np.dot(batch_images, self.w1) + self.b1
        a1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)  # 用softmax函数进行分类。

        batch_labels = batch_labels.astype(int)
        """
        # 计算交叉熵损失函数
        num_samples = batch_images.shape[0]
        loss = -np.sum(np.log(a2[range(num_samples), batch_labels])) / num_samples
        """

        # 计算均方误差损失函数
        num_samples = batch_images.shape[0]
        # print('**************')
        # print(range(num_samples) - batch_labels)
        # print('***************')
        # print(a2)
        # print('#############')
        if batch_labels.size == 1:
            loss = np.sum(np.square(a2[0][range(num_samples) - batch_labels])) / (2 * num_samples)
        else:
            loss = np.sum(np.square(a2[range(num_samples) - batch_labels])) / (2 * num_samples)


        # 反向传播
        dz2 = a2
        dz2[range(num_samples), batch_labels] -= 1
        dz2 /= num_samples

        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dz1 = np.dot(dz2, self.w2.T) * a1 * (1 - a1)
        dw1 = np.dot(batch_images.T, dz1)
        db1 = np.sum(dz1, axis=0)
        return loss, dw1, dw2, db1, db2

    # 测试模型
    def test(self, test_images, test_labels):
        z1 = np.dot(test_images, self.w1) + self.b1
        a1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)
        # 预测结果
        predictions = np.argmax(a2, axis=1)
        accuracy = np.mean(predictions == test_labels)
        print(f'Test Accuracy: {accuracy:.4f}')

    # 使用mini-GD梯度下降法
    def mini_GD(self, epochs, train_images, batch_size, train_labels):
        for epoch in range(epochs):
            for i in range(0, len(train_images), batch_size):
                # 选择一个小批量样本
                batch_images = train_images[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                loss, dw1, dw2, db1, db2 = self.Backpropagation(batch_images, batch_labels)
                # 更新权重和偏置
                self.w1 -= self.lr * dw1
                self.b1 -= self.lr * db1
                self.w2 -= self.lr * dw2
                self.b2 -= self.lr * db2
                # 打印每个epoch的损失
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    # 使用batch_GD梯度下降法
    def batch_GD(self, epochs, train_images, train_labels):
        for epoch in range(epochs):
            loss, dw1, dw2, db1, db2 = self.Backpropagation(train_images, train_labels)
            # 更新权重和偏置
            self.w1 -= self.lr * dw1
            self.b1 -= self.lr * db1
            self.w2 -= self.lr * dw2
            self.b2 -= self.lr * db2
            # 打印每个epoch的损失
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    # 使用SGD梯度下降法
    def SGD(self, epochs, train_images, train_labels):
        for epoch in range(epochs):
            for i in range(0, len(train_images)):
                x_images = train_images[i:i + 1]
                y_labels = train_labels[i:i + 1]
                loss, dw1, dw2, db1, db2 = self.Backpropagation(x_images, y_labels)
                # 更新权重和偏置
                self.w1 -= self.lr * dw1
                self.b1 -= self.lr * db1
                self.w2 -= self.lr * dw2
                self.b2 -= self.lr * db2
                # 打印每个epoch的损失
                print(f'Epoch {i}/{epoch}, Loss: {loss:.4f}')


inputnodes = 784
hiddennodes = 16
outputnodes = 10
epochs = 100
learning_rate = 0.1
batch_size = 64
n = neuralNetwork(inputnodes, hiddennodes, outputnodes, learning_rate)
# n.mini_GD(epochs, X_train, batch_size, y_train)
# n.batch_GD(epochs, X_train, y_train)
n.SGD(epochs, X_train, y_train)
n.test(X_test, y_test)
