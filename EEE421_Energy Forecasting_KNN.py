# -*- codeing = utf-8 -*-
# @Time : 2023/11/5 14:47
# @Author : ZhangQH
# @File : EEE421_Energy Forecasting_KNN.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

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

# 创建KNN模型
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

# 保存模型
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

# 加载已保存的模型
with open('knn_model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)
# 计算准确率和R2分数
accuracy = knn_model.score(X_test, y_test)
r2 = r2_score(y_test, y_pred)

# 计算均方误差
rmse = mean_squared_error(y_test, y_pred, squared=False)

# 输出准确率和R2分数
print("Accuracy:", accuracy)
print("R2 Score:", r2)
print("RMSE:", rmse)

# 绘制测试集的真实值和预测值图像
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('KNN Prediction Chart')
plt.legend()
plt.show()

#对“Data for group projects”进行预测

new_data = pd.read_excel(r'D:\huahua\Master_XJTLU\2023-2024-S1\EEE421\program\data_new\published_group_data1.xlsx')
# 提取特征变量
X_new = new_data[['HOT_WATER_ENERGY_EFF', 'FLOOR_ENERGY_EFF', 'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF',
                  'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']]

# 加载已训练好的线性回归模型
with open('knn_model.pkl', 'rb') as f:
    model_knn = pickle.load(f)

# 进行预测
y_pred = model_knn.predict(X_new)

# 输出预测结果
print(y_pred)