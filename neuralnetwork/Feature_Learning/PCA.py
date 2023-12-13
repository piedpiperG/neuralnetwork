from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import numpy as np

# 加载MNIST数据集的一部分
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"][:9000], mnist["target"][:9000]  # 只使用前3000个样本

# 将数据类型转换为整型
y = y.astype(np.uint8)

# 使用PCA进行降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 调整Kernel PCA参数
kernel_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)  # 调整gamma值
X_kernel_pca = kernel_pca.fit_transform(X)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.5)
plt.title("PCA")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_kernel_pca[:, 0], X_kernel_pca[:, 1], c=y, cmap="viridis", alpha=0.5)
plt.title("Kernel PCA")
plt.colorbar()

plt.show()
