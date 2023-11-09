# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:16:04 2023

@author: Alienware
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("iris.txt", sep=' ')  # 使用空格作为分隔符

# data.fillna(data.median(), inplace=True)

# 分割特征和标签
X = data.iloc[:, :3]  # 特征列
y = data.iloc[:, -2]  # 标签列
# print(X)
# print(y)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练和测试逻辑回归
logistic_model = LogisticRegression()
logistic_model.fit(X_train.astype("int"), y_train.astype("int"))
logistic_accuracy = logistic_model.score(X_test.astype("int"), y_test.astype("int"))

# 训练和测试决策树
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train.astype("int"), y_train.astype("int"))
tree_accuracy = tree_model.score(X_test.astype("int"), y_test.astype("int"))

print("逻辑回归准确率:", logistic_accuracy)
print("决策树准确率:", tree_accuracy)
