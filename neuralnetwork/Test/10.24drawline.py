import numpy as np
import matplotlib.pyplot as plt
# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
# 创建输入数据
x = np.linspace(-5, 5, 1000)
# 计算不同激活函数的输出
sigmoid_y = sigmoid(x)
tanh_y = tanh(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
elu_y = elu(x)
# 绘制图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x, sigmoid_y, label='Sigmoid', color='r')
plt.plot(x, tanh_y, label='Tanh', color='g')
plt.plot(x, relu_y, label='ReLU', color='b')
plt.plot(x, leaky_relu_y, label='Leaky ReLU', color='c')
plt.plot(x, elu_y, label='ELU', color='m')  # 添加ELU函数的曲线
plt.ylim(-1.2, 1.2)
plt.title('Activation Functions')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, sigmoid_y, label='Sigmoid', color='r')
plt.plot(x, tanh_y, label='Tanh', color='g')
plt.plot(x, relu_y, label='ReLU', color='b')
plt.plot(x, leaky_relu_y, label='Leaky ReLU', color='c')
plt.plot(x, elu_y, label='ELU', color='m')  # 添加ELU函数的曲线
plt.ylim(-0.2, 6)
plt.title('Zoomed-in View')
plt.legend()
plt.show()
