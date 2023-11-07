# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:40:00 2023

@author: Alienware
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 加载数据
data = pd.read_csv("D:\作业\大作业\统计学习方法\eg6.1.xls", encoding='GB2312')


# 数据预处理，如缺失值处理、标准化等

# 创建PCA对象并拟合数据
pca = PCA(n_components=2)  # 设置保留的主成分个数
pca.fit(data)

# 计算主成分得分
scores = pca.transform(data)

# 绘制得分图
plt.scatter(scores[:, 0], scores[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scores Plot')
plt.show()

# 绘制载荷图
loadings = pca.components_.T
plt.bar(range(data.shape[1]), loadings[:, 0], alpha=0.5, label='PC1')
plt.bar(range(data.shape[1]), loadings[:, 1], alpha=0.5, label='PC2')
plt.xlabel('Features')
plt.ylabel('Loadings')
plt.title('Loadings Plot')
plt.legend()
plt.show()