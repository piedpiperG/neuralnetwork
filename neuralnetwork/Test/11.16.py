# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:40:56 2023

@author: Alienware
"""

import pandas as pd
import statsmodels.api as sm

data = pd.read_excel("ex8.xlsx")

# 填充 X2 中的空值为 X2 的平均值
X2 = pd.to_numeric(data['5、多元统计分析课程我预期的成绩'], errors='coerce')
X2_mean = X2.mean()
X2 = X2.fillna(X2_mean)

X1 = data['3、我的数分1成绩+数分2成绩']
X1 = sm.add_constant(X1)  # 加入常数项，便于计算截距

X2 = pd.Series(X2)  # 将X2转换为Series类型

X = pd.concat([X1, X2], axis=1)  # 拼接两个自变量

Y = data['序号']

model = sm.OLS(Y, X)  # 构建最小二乘回归模型
results = model.fit()  # 拟合模型

print(results.summary())
