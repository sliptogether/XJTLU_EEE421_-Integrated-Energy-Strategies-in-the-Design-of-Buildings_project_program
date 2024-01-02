# -*- codeing = utf-8 -*-
# @Time : 2023/11/5 15:00
# @Author : ZhangQH
# @File : EEE421_Energy Foresting_Compare.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据集
data = pd.read_csv(r'D:\huahua\Master_XJTLU\2023-2024-S1\EEE421\program\data_new\data.csv')

# 提取特征和目标变量
col_water = ['HOT_WATER_ENERGY_EFF']
col_floor = ['FLOOR_ENERGY_EFF']
col_windows = ['WINDOWS_ENERGY_EFF']
col_wall = ["WALLS_ENERGY_EFF"]
col_target = ["CURRENT_ENERGY_EFFICIENCY"]
col_roof = ["ROOF_ENERGY_EFF"]
col_heat = ["MAINHEAT_ENERGY_EFF"]
col_heat_cold = ["MAINHEATC_ENERGY_EFF"]
col_light = ["LIGHTING_ENERGY_EFF"]

X = data[col_water + col_floor + col_windows + col_wall + col_roof + col_heat + col_heat_cold + col_light]
y = data[col_target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)
lr_accuracy = lr_model.score(X_test, y_test)

# 创建支持向量机（SVM）模型
svm_model = SVR()
svm_model.fit(X_train, y_train)
y_svm_pred = svm_model.predict(X_test)
svm_accuracy = svm_model.score(X_test, y_test)

# 创建随机森林模型
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)
rf_accuracy = rf_model.score(X_test, y_test)

# 创建KNN模型
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_knn_pred = knn_model.predict(X_test)
knn_accuracy = knn_model.score(X_test, y_test)

# 计算均方误差
lr_rmse = mean_squared_error(y_test, y_lr_pred, squared=False)
svm_rmse = mean_squared_error(y_test, y_svm_pred, squared=False)
rf_rmse = mean_squared_error(y_test, y_rf_pred, squared=False)
knn_rmse = mean_squared_error(y_test, y_knn_pred, squared=False)

# 绘制预测值对比图
plt.scatter(y_test, y_lr_pred, label='Linear Regression')
plt.scatter(y_test, y_svm_pred, label='SVM')
plt.scatter(y_test, y_rf_pred, label='Random Forest')
plt.scatter(y_test, y_knn_pred, label='KNN')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Models Comparison')
plt.legend()
plt.show()

# 输出均方误差对比
print("Linear Regression RMSE:", lr_rmse)
print("SVM RMSE:", svm_rmse)
print("Random Forest RMSE:", rf_rmse)
print("KNN RMSE:", knn_rmse)

# 输出准确率
print("Linear Regression Accuracy:", lr_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("KNN Accuracy:", knn_accuracy)