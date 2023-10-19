import numpy as np
from scipy.io import loadmat


# 加载数据文件
def loadFile():
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

    return x_train, y_train, x_test, y_test


# 定义 softmax 函数
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# 初始化权重矩阵和偏置向量
def initialize_parameters(input_size, output_size):
    W = np.random.randn(input_size, output_size) * 0.001
    b = np.zeros(output_size)
    return W, b


# 定义交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    y_true = y_true.astype(int)  # 强制转换为整数类型
    return -np.sum(np.log(y_pred[np.arange(m), y_true])) / m



# 定义准确率计算函数
def accuracy(y_true, y_pred):
    return np.mean(y_true == np.argmax(y_pred, axis=1))


# 训练 softmax 分类器
def train_softmax_classifier(x_train, y_train, x_test, y_test, learning_rate, num_epochs, batch_size):
    input_size = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    W, b = initialize_parameters(input_size, num_classes)

    m = x_train.shape[0]
    for epoch in range(num_epochs):
        for i in range(0, m, batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            y_batch = y_batch.astype(int)  # 强制转换为整数类型
            # 计算预测值
            scores = np.dot(x_batch, W) + b
            y_pred = softmax(scores)
            # 计算损失
            loss = cross_entropy_loss(y_batch, y_pred)
            # 计算梯度
            grad = y_pred
            grad[np.arange(x_batch.shape[0]), y_batch] -= 1
            grad /= batch_size

            dW = np.dot(x_batch.T, grad)
            db = np.sum(grad, axis=0)

            # 更新权重和偏置
            W -= learning_rate * dW
            b -= learning_rate * db

        # 在每个 epoch 结束后打印损失和准确率
        y_test_pred = softmax(np.dot(x_test, W) + b)
        test_loss = cross_entropy_loss(y_test, y_test_pred)
        test_accuracy = accuracy(y_test, y_test_pred)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}")


# 设置超参数
learning_rate = 0.1
num_epochs = 100
batch_size = 128

# 加载数据集
x_train, y_train, x_test, y_test = loadFile()

# 训练 softmax 分类器
train_softmax_classifier(x_train, y_train, x_test, y_test, learning_rate, num_epochs, batch_size)
