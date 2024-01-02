# -*- codeing = utf-8 -*-
# @Time : 2023/11/4 12:48
# @Author : ZhangQH
# @File : EEE421_Energy Foresting_Compare.py
# @Software: PyCharm
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle


# 读取数据集
data = pd.read_csv(r"D:\huahua\Master_XJTLU\2023-2024-S1\EEE421\program\data_new\data.csv")
data.shape
data.head()
data.info()

# 多重共线性
# VIF

feature_names = ['HOT_WATER_ENERGY_EFF','WINDOWS_ENERGY_EFF','WALLS_ENERGY_EFF',
                   'ROOF_ENERGY_EFF','MAINHEAT_ENERGY_EFF','MAINHEATC_ENERGY_EFF','LIGHTING_ENERGY_EFF']
X=data[feature_names]
y=data.POSTCODE

# 计算VIF

vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns)

# 展示VIF结果

index = X.columns.tolist()
vif_df = pd.DataFrame(vif, index = index,
                      columns = ['vif']).sort_values(by = 'vif', ascending=False)
vif_df
print(vif_df)

# 提取特征和目标变量

X = data[['HOT_WATER_ENERGY_EFF', 'FLOOR_ENERGY_EFF', 'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF',
          'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']]
y = data['CURRENT_ENERGY_EFFICIENCY']

# 拆分数据集为训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model_LR = LinearRegression()

# 在训练集上训练模型
model_LR.fit(X_train, y_train)

# 保存模型
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model_LR, f)

# 加载已保存的模型
with open('linear_regression_model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)

# 在测试集上进行预测
y_pred = model_LR.predict(X_test)

# 评估模型

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Square Error (MSE):', mse)
print('Coefficient Ofdetermination (R2):', r2)

# 绘制线性回归图

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Value')
plt.ylabel('Forecast Value')
plt.title('Linear Regression Prediction Chart')
plt.show()

# 展示准确率
accuracy = model_LR.score(X_test, y_test)
print('Accuracy:', accuracy)

#对“Data for group projects”进行预测

new_data = pd.read_excel(r'D:\huahua\Master_XJTLU\2023-2024-S1\EEE421\program\data_new\published_group_data1.xlsx')
# 提取特征变量
X_new = new_data[['HOT_WATER_ENERGY_EFF', 'FLOOR_ENERGY_EFF', 'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF',
                  'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']]

# 加载已训练好的线性回归模型
with open('linear_regression_model.pkl', 'rb') as f:
    model_LR = pickle.load(f)

# 进行预测
y_pred = model_LR.predict(X_new)

# 输出预测结果
print(y_pred)
