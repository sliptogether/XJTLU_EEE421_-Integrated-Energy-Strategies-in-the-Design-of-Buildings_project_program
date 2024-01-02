# -*- codeing = utf-8 -*-
# @Time : 2023/11/5 14:18
# @Author : ZhangQH
# @File : EEE421_Energy Forecasting_SVM.py
# @Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.preprocessing import StandardScaler
# 读取数据集
data = pd.read_csv(r'D:\huahua\Master_XJTLU\2023-2024-S1\EEE421\program\data_new\data.csv')

# 提取特征和目标变量
col_target = ["CURRENT_ENERGY_EFFICIENCY"]
col_water = ['HOT_WATER_ENERGY_EFF']
col_floor = ['FLOOR_ENERGY_EFF']
col_windows = ['WINDOWS_ENERGY_EFF']
col_wall = ["WALLS_ENERGY_EFF"]
col_roof = ["ROOF_ENERGY_EFF"]
col_heat = ["MAINHEAT_ENERGY_EFF"]
col_heat_cold = ["MAINHEATC_ENERGY_EFF"]
col_light = ["LIGHTING_ENERGY_EFF"]

X_features = data[col_water + col_floor + col_windows + col_wall + col_roof + col_heat + col_heat_cold + col_light]
y = data[col_target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建SVM模型
model_SVM = SVR()

# 拟合训练集
model_SVM.fit(X_train_scaled, y_train)

# 保存模型
with open('SVM_model.pkl', 'wb') as f:
    pickle.dump(model_SVM, f)

# 加载已保存的模型
with open('SVM_model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)

# 在训练集上进行预测
y_train_pred = model_SVM.predict(X_train_scaled)

# 在测试集上进行预测
y_test_pred = model_SVM.predict(X_test_scaled)

# 计算准确率和R2分数
train_accuracy = model_SVM.score(X_train_scaled, y_train)
test_accuracy = model_SVM.score(X_test_scaled, y_test)

# 计算均方误差
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# 输出准确率、R2分数和均方误差
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# 绘制测试集的真实值和预测值图像
plt.scatter(y_test, y_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('SVM Prediction Chart')
plt.show()

#对“Data for group projects”进行预测

new_data = pd.read_excel(r'D:\huahua\Master_XJTLU\2023-2024-S1\EEE421\program\data_new\published_group_data1.xlsx')
# 提取特征变量

X_new = new_data[['HOT_WATER_ENERGY_EFF', 'FLOOR_ENERGY_EFF', 'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF',
                  'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']]

# 加载已训练好的线性回归模型
with open('SVM_model.pkl', 'rb') as f:
    model_SVM = pickle.load(f)

group_features_scaled = scaler.transform(X_new)
# 进行预测
y_pred = model_SVM.predict(group_features_scaled)

# 输出预测结果
print(y_pred)