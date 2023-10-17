from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitialize import initialise
from Prediction import predict
from scipy.optimize import minimize

# 加载数据文件
data = loadmat('mnist-original.mat')

# 提取数据的特征矩阵，并进行转置
X = data['data']
X = X.transpose()

# 然后将特征除以255，重新缩放到[0,1]的范围内，以避免在计算过程中溢出
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

input_layer_size = 784  # 图片大小为 (28 X 28) px 所以设置784个特征
hidden_layer_size = 100
num_labels = 10  # 拥有十个标准为 [0, 9] 十个数字

# 随机初始化 Thetas
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)  # 输入层和隐藏层之间的权重
initial_Theta2 = initialise(num_labels, hidden_layer_size)  # 隐藏层和输出层之间的权重

# 设置神经网络的参数
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
lambda_reg = 0.1  # 避免过拟合
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# 设置最小化函数的迭代次数
maxiter = 100
# 调用最小化函数来计算权重
results = minimize(neural_network, x0=initial_nn_params, args=myargs,
                   options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)
nn_params = results["x"]  # 获得训练之后的权重

# 重新分割，获得三个层次之间两两的权重
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
    hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                    (num_labels, hidden_layer_size + 1))  # shape = (10, 101)

# 测试集的准确度
pred = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100)))

# 训练集的准确度
pred = predict(Theta1, Theta2, X_train)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))

# 模型的准确率(Precision)
true_positive = 0
for i in range(len(pred)):
    if pred[i] == y_train[i]:
        true_positive += 1
false_positive = len(y_train) - true_positive
print('Precision =', true_positive / (true_positive + false_positive))

# 将Theta参数保存在txt文件中，用作后续程序识别
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')
