import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义损失函数
def mse_loss(w1, w2):
    # 这里假设输入为1，输出为1
    input_data = 1
    actual_output = 1
    hidden_output = 1 / (1 + np.exp(-(w1 * input_data)))  # Sigmoid激活函数
    predicted_output = w2 * hidden_output
    return 0.5 * (predicted_output - actual_output) ** 2

def cross_entropy_loss(w1, w2):
    input_data = 1
    actual_output = 1
    hidden_output = 1 / (1 + np.exp(-(w1 * input_data)))  # Sigmoid激活函数
    predicted_output = w2 * hidden_output
    return - (actual_output * np.log(predicted_output) + (1 - actual_output) * np.log(1 - predicted_output))

# 创建参数空间
w1_range = np.linspace(-5, 5, 100)
w2_range = np.linspace(-5, 5, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

# 计算损失函数值
MSE_loss = np.zeros_like(W1)
CE_loss = np.zeros_like(W2)

for i in range(len(w1_range)):
    for j in range(len(w2_range)):
        MSE_loss[i][j] = mse_loss(W1[i][j], W2[i][j])
        CE_loss[i][j] = cross_entropy_loss(W1[i][j], W2[i][j])

# 绘制三维图像
fig = plt.figure(figsize=(15, 6))

# 均方误差损失函数图像
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1, W2, MSE_loss, cmap='viridis')
ax1.set_title('MSE Loss')
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('Loss')

# 交叉熵损失函数图像
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(W1, W2, CE_loss, cmap='viridis')
ax2.set_title('Cross-Entropy Loss')
ax2.set_xlabel('w1')
ax2.set_ylabel('w2')
ax2.set_zlabel('Loss')

plt.show()
