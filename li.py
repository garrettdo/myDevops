# -*- coding: utf-8 -*-
# @Author  : li
# @Time    : Thu Jun 17 10:55:50 CST 2021

# 加载函数库
import openpyxl
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from IPython.utils.tests.test_wildcard import q
from docutils.nodes import inline
# from statsmodels.genmod.tests.gee_simulation_check import q
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import warnings
from statsmodels.tsa.stattools import adfuller as ADF

# 显示配置
# %matplotlib inline
plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = pd.read_excel('SystemLoadAndDiskData.xls')

# 前五行数据预览
data.head()

# 基本属性信息
data.info()

# 数值属性的中心趋势度量1
data.describe()

# 　数值属性的简单图示
plt.figure(figsize=(12, 6))
data['VALUE'].plot()
plt.show()

# 获取C盘磁盘容量 (注意转义字符的使用)
disk_storage = data.loc[(data['ENTITY'] == 'C:\\') & (data['TARGET_ID'] == 183), ['COLLECTTIME', 'VALUE']]
disk_storage.set_index('COLLECTTIME', inplace=True)
disk_storage.columns = ['VALUE_C']
disk_storage.head()

# 获取D盘磁盘容量 (注意转义字符的使用)
disk_storage['VALUE_D'] = data.loc[(data['ENTITY'] == 'D:\\') & (data['TARGET_ID'] == 183), ['VALUE']].values
disk_storage.head()

# 简单的图示
plt.figure(figsize=(12, 6))
disk_storage[["VALUE_C", "VALUE_D"]].plot()
plt.ylim([0.3e+8, 2e+8])
plt.grid()
plt.show()

# 获取C盘已使用情况
disk_usage = data.loc[(data['ENTITY'] == 'C:\\') & (data['TARGET_ID'] == 184), ['COLLECTTIME', 'VALUE']]
disk_usage.set_index('COLLECTTIME', inplace=True)
disk_usage.columns = ['VALUE_C']
disk_usage.head()

# 获取D盘已使用情况
disk_usage['VALUE_D'] = data.loc[(data['ENTITY'] == 'D:\\') & (data['TARGET_ID'] == 184), ['VALUE']].values
disk_usage.head()

# 简单的图示
plt.figure(figsize=(12, 6))
disk_usage["VALUE_C"].plot(color='blue', linewidth=2.0, linestyle='-', marker='o', markersize=12, markerfacecolor='r')
plt.title("C盘已使用情况")
plt.xlabel('日期')
plt.xticks(disk_usage.index, rotation=30)
plt.grid()
plt.show()

# 简单的图示
plt.figure(figsize=(12, 6))
disk_usage["VALUE_D"].plot(color='blue', linewidth=2.0, linestyle='-', marker='o', markersize=12, markerfacecolor='r')
plt.title("D盘已使用情况")
plt.xlabel('日期')
plt.xticks(disk_usage.index, rotation=30)
plt.grid()
plt.show()

disk_usage.info()
disk_usage.describe()

# 自相关与偏相关 - C盘
C_usage = disk_usage['VALUE_C']
fig = plt.figure(figsize=(12, 16))
ax1 = fig.add_subplot(411)
sm.graphics.tsa.plot_acf(C_usage, lags=22, ax=ax1)
ax1.set_title('自相关图 - C盘已使用存储空间的时间序列')

ax2 = fig.add_subplot(412)
sm.graphics.tsa.plot_pacf(C_usage, lags=22, ax=ax2)
ax2.set_title('偏自相关图 - C盘已使用存储空间的时间序列')

# 一阶差分后去空值取自相关系数
C_usage_diff = C_usage.diff(1).dropna()
ax3 = fig.add_subplot(413)
sm.graphics.tsa.plot_acf(C_usage_diff, lags=22, ax=ax3)
ax3.set_title('自相关图 - 一阶差分')

ax4 = fig.add_subplot(414)
sm.graphics.tsa.plot_pacf(C_usage_diff, lags=22, ax=ax4)
ax4.set_title('偏自相关图 - 一阶差分')

plt.tight_layout()
plt.show()

# 自相关与偏相关 - D盘
D_usage = disk_usage['VALUE_D']
fig = plt.figure(figsize=(12, 16))
ax1 = fig.add_subplot(411)
sm.graphics.tsa.plot_acf(D_usage, lags=22, ax=ax1)
ax1.set_title('自相关图 - D盘已使用存储空间的时间序列')

ax2 = fig.add_subplot(412)
sm.graphics.tsa.plot_pacf(D_usage, lags=22, ax=ax2)
ax2.set_title('偏自相关图 - D盘已使用存储空间的时间序列')

# 一阶差分后去空值取自相关系数
D_usage_diff = D_usage.diff(1).dropna()
ax3 = fig.add_subplot(413)
sm.graphics.tsa.plot_acf(D_usage_diff, lags=22, ax=ax3)
ax3.set_title('自相关图 - 一阶差分')

ax4 = fig.add_subplot(414)
sm.graphics.tsa.plot_pacf(D_usage_diff, lags=22, ax=ax4)
ax4.set_title('偏自相关图 - 一阶差分')

plt.tight_layout()
plt.show()

# ADF检验
# 注意：预留最后5个数字用于对模型性能进行评估
data = disk_usage.iloc[:len(disk_usage) - 5]

# 平稳性测试函数

diff = 0
adf = ADF(data['VALUE_C'])

# adf[1]为p值，p值小于0.05认为是平稳的
while adf[1] >= 0.05:
    diff = diff + 1
    adf = ADF(data['VALUE_C'].diff(diff).dropna())

print('C盘已使用存储空间的时间序列经过%s阶差分后归于平稳，p值为%s' % (diff, adf[1]))

diff = 0
adf = ADF(data['VALUE_D'])

# adf[1]为p值，p值小于0.05认为是平稳的
while adf[1] >= 0.05:
    diff = diff + 1
    adf = ADF(data['VALUE_D'].diff(diff).dropna())

print('D盘已使用存储空间的时间序列经过%s阶差分后归于平稳，p值为%s' % (diff, adf[1]))

# 白噪声检验
# LB统计量

[[lb], [p]] = acorr_ljungbox(data['VALUE_C'], lags=1)
if p < 0.05:
    print('C盘已使用存储空间的时间序列为 非白噪声序列，对应的p值为：%s' % p)
else:
    print('C盘已使用存储空间的时间序列为 白噪声序列，对应的p值为：%s' % p)

[[lb], [p]] = acorr_ljungbox(data['VALUE_C'].diff(1).dropna(), lags=1)
if p < 0.05:
    print('其一阶差分序列为 非白噪声序列，对应的p值为：%s' % p)
else:
    print('其一阶差分为 白噪声序列，对应的p值为：%s' % p)

[[lb], [p]] = acorr_ljungbox(data['VALUE_D'], lags=1)
if p < 0.05:
    print('D盘已使用存储空间的时间序列为 非白噪声序列，对应的p值为：%s' % p)
else:
    print('D盘已使用存储空间的时间序列为 白噪声序列，对应的p值为：%s' % p)

[[lb], [p]] = acorr_ljungbox(data['VALUE_D'].diff(1).dropna(), lags=1)
if p < 0.05:
    print('其一阶差分序列为 非白噪声序列，对应的p值为：%s' % p)
else:
    print('其一阶差分为 白噪声序列，对应的p值为：%s' % p)


# 获得模型参数 - 最佳p/q值
def arima_para(time_series, d=1):
    """ 对于给定的平稳时间序列，基于BIC指标的ARIA模型最佳p/q值

    d - 差分阶数；默认为1阶
    """

    # 定阶
    paar = int(len(time_series) / 10)  # 一般阶数不超过length/10
    qmark = int(len(time_series) / 10)

    bic_matrix = []  # bic矩阵
    for pa in range(paar + 1):
        tmp = []
        for qa in range(qmark + 1):
            try:
                tmp.append(ARIMA(time_series, (pa, d, qa)).fit().bic)
            except:
                tmp.append(None)

        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    pa, qa = bic_matrix.stack().astype('float64').idxmin()

    return pa, qa


warnings.filterwarnings("ignore")

x_C = data['VALUE_C']
p_C, q_C = arima_para(x_C)
print('C盘已使用存储空间的时间序列： ARIA模型最佳p值和q值为:%s、%s' % (p_C, q_C))

# 计算模型预测输出
dataArm_C = ARIMA(x_C, (p_C, 1, q_C)).fit()
x_pred = dataArm_C.predict(typ='levels')

# 获得原始时间序列的残差
x_err = (x_pred - x_C).dropna()

# 检查残差序列是否是白噪声
lagnum = 12
lb, p = acorr_ljungbox(x_err, lags=lagnum)

# p值小于0.05，认为是非白噪声
h = (p < 0.05).sum()
if h > 0:
    print('C盘已使用存储空间的时间序列: 模型ARIA（%s,1,%s）不符合白噪声检验' % (p, q))
else:
    print('C盘已使用存储空间的时间序列: 模型ARIA（%s,1,%s）符合白噪声检验' % (p, q))


# 计算模型预测输出
# def x_D(args):
#     pass
#
#
# def p_D(args):
#     pass
#
#
# def q_D(args):
#     pass


# dataArm_D = ARIMA(x_D, (arima_para(D_usage), 1, arima_para(D_usage_diff))).fit()
# x_pred = dataArm_D.predict(typ='levels')

x_D = data['VALUE_D']
p_D, q_D = arima_para(x_D)
print('D盘已使用存储空间的时间序列： ARIA模型最佳p值和q值为:%s、%s' % (p_D, q_D))

dataArm_D = ARIMA(x_D, (p_D, 1, q_D)).fit()
x_pred = dataArm_D.predict(typ='levels')

# 获得原始时间序列的残差
x_err = (x_pred - x_D).dropna()

# 检查残差序列是否是白噪声
lagnum = 10
lb, p = acorr_ljungbox(x_err, lags=lagnum)

# p值小于0.05，认为是非白噪声
h = (p < 0.05).sum()
if h > 0:
    print('D盘已使用存储空间的时间序列: 模型ARIA（%s,1,%s）不符合白噪声检验' % (p, q))
else:
    print('D盘已使用存储空间的时间序列: 模型ARIA（%s,1,%s）符合白噪声检验' % (p, q))

# C盘使用情况预测
y_forecast = dataArm_C.forecast(5)[0]
y_true = disk_usage.iloc[len(disk_usage) - 5:]['VALUE_C']

comp_C = pd.DataFrame({"C盘使用预测值": y_forecast, "C盘使用实际值": y_true})
comp_C = comp_C.applymap(lambda x: '%.2f' % x)
comp_C

# 性能评估
abs_ = (y_forecast - y_true).abs()
mae_ = abs_.mean()
rmse_ = ((abs_ ** 2).mean()) ** 0.05
mape_ = (abs_ / y_true).mean()

print('C盘已使用存储空间的时间序列：\n\n平均绝对误差为：%0.4f,\n均方根误差为：%0.4f,\n平均绝对百分误差为：%0.6f' % (mae_, rmse_, mape_))

# D盘使用情况预测
y_forecast = dataArm_D.forecast(5)[0]
y_true = disk_usage.iloc[len(disk_usage) - 5:]['VALUE_D']

comp_D = pd.DataFrame({"D盘使用预测值": y_forecast, "D盘使用实际值": y_true})
comp_D = comp_D.applymap(lambda x: '%.2f' % x)
comp_D

# 性能评估
abs_ = (y_forecast - y_true).abs()
mae_ = abs_.mean()
rmse_ = ((abs_ ** 2).mean()) ** 0.05
mape_ = (abs_ / y_true).mean()

print('D盘已使用存储空间的时间序列：\n\n平均绝对误差为：%0.4f,\n均方根误差为：%0.4f,\n平均绝对百分误差为：%0.6f' % (mae_, rmse_, mape_))
