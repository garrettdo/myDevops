
# 0 数据探索

## 加载函数库

```bash
import openpyxl
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from IPython.utils.tests.test_wildcard import q
from docutils.nodes import inline
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import warnings
from statsmodels.tsa.stattools import adfuller as ADF
```

## 显示配置

```bash
plt.rcParams['font.family'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
data = pd.read_excel('SystemLoadAndDiskData.xls')
```

## 前五行数据预览

```bash
data.head()
```

## 基本属性信息

```bash
data.info()
```

## 数值属性的中心趋势度量1

```bash
data.describe()
```

##　数值属性的简单图示

+ myplot-00.png

```bash
plt.figure(figsize=(12,6))
data['VALUE'].plot()
plt.show()
```

## 获取C盘磁盘容量 (注意转义字符的使用)

```bash
disk_storage = data.loc[(data['ENTITY'] == 'C:\\') & (data['TARGET_ID'] == 183),['COLLECTTIME','VALUE']]
disk_storage.set_index('COLLECTTIME',inplace = True)
disk_storage.columns = ['VALUE_C']
disk_storage.head()
```

## 获取D盘磁盘容量 (注意转义字符的使用)

```bash
disk_storage['VALUE_D'] = data.loc[(data['ENTITY'] == 'D:\\') & (data['TARGET_ID'] == 183), ['VALUE']].values
disk_storage.head()
```

## 简单的图示

+ myplot-01.png

```bash
plt.figure(figsize=(12,6))
disk_storage[["VALUE_C", "VALUE_D"]].plot()
plt.ylim([0.3e+8, 2e+8])
plt.grid()
plt.show()
```

## 获取C盘已使用情况

+ myplot-02.png

```bash
disk_usage = data.loc[(data['ENTITY'] == 'C:\\') & (data['TARGET_ID'] == 184),['COLLECTTIME','VALUE']]
disk_usage.set_index('COLLECTTIME',inplace = True)
disk_usage.columns = ['VALUE_C']
disk_usage.head()
```

# 获取D盘已使用情况

+ myplot-03.png

```bash
disk_usage['VALUE_D'] = data.loc[(data['ENTITY'] == 'D:\\') & (data['TARGET_ID'] == 184),['VALUE']].values
disk_usage.head()
```

## 简单的图示

```bash
plt.figure(figsize=(12,6))
disk_usage["VALUE_C"].plot(color='blue',linewidth=2.0,linestyle='-',marker='o',markersize=12,markerfacecolor='r')
plt.title("C盘已使用情况")
plt.xlabel('日期')
plt.xticks(disk_usage.index, rotation = 30)
plt.grid()
plt.show()
```

## 简单的图示

```bash
plt.figure(figsize=(12,6))
disk_usage["VALUE_D"].plot(color='blue',linewidth=2.0,linestyle='-',marker='o',markersize=12,markerfacecolor='r')
plt.title("D盘已使用情况")
plt.xlabel('日期')
plt.xticks(disk_usage.index, rotation = 30)
plt.grid()
plt.show()
```


# 1. 数据预处理

参考资料：
 - [时间序列（一）：平稳性、自相关函数与LB检验](https://zhuanlan.zhihu.com/p/54153963)
 - [如何理解自相关和偏自相关图](https://blog.csdn.net/Yuting_Sunshine/article/details/95317735)
 
## 自相关与偏相关 - C盘

+ myplot-04.png

```bash
import statsmodels.api as sm

C_usage = disk_usage['VALUE_C']
fig = plt.figure(figsize = (12,16))
ax1 = fig.add_subplot(411)
sm.graphics.tsa.plot_acf(C_usage, lags = 40, ax = ax1)
ax1.set_title('自相关图 - C盘已使用存储空间的时间序列')

ax2 = fig.add_subplot(412)
sm.graphics.tsa.plot_pacf(C_usage, lags = 40, ax = ax2)
ax2.set_title('偏自相关图 - C盘已使用存储空间的时间序列')
```

### 一阶差分后去空值取自相关系数

```bash
C_usage_diff = C_usage.diff(1).dropna() 
ax3 = fig.add_subplot(413)
sm.graphics.tsa.plot_acf(C_usage_diff, lags = 40, ax = ax3)
ax3.set_title('自相关图 - 一阶差分')

ax4 = fig.add_subplot(414)
sm.graphics.tsa.plot_pacf(C_usage_diff, lags = 40, ax = ax4)
ax4.set_title('偏自相关图 - 一阶差分')

plt.tight_layout()
plt.show()

```


## 自相关与偏相关 - D盘

+ myplot-05.png

```bash
D_usage = disk_usage['VALUE_D']
fig = plt.figure(figsize = (12,16))
ax1 = fig.add_subplot(411)
sm.graphics.tsa.plot_acf(D_usage, lags = 40, ax = ax1)
ax1.set_title('自相关图 - D盘已使用存储空间的时间序列')

ax2 = fig.add_subplot(412)
sm.graphics.tsa.plot_pacf(D_usage, lags = 40, ax = ax2)
ax2.set_title('偏自相关图 - D盘已使用存储空间的时间序列')
```

### 一阶差分后去空值取自相关系数
```bash
D_usage_diff = D_usage.diff(1).dropna() 
ax3 = fig.add_subplot(413)
sm.graphics.tsa.plot_acf(D_usage_diff, lags = 40, ax = ax3)
ax3.set_title('自相关图 - 一阶差分')

ax4 = fig.add_subplot(414)
sm.graphics.tsa.plot_pacf(D_usage_diff, lags = 40, ax = ax4)
ax4.set_title('偏自相关图 - 一阶差分')

plt.tight_layout()
plt.show()
```


# ADF检验  

+ 注意：预留最后5个数字用于对模型性能进行评估

参考资料：
 - [平稳随机过程](https://baike.baidu.com/item/%E5%B9%B3%E7%A8%B3%E9%9A%8F%E6%9C%BA%E8%BF%87%E7%A8%8B)

```bash
data = disk_usage.iloc[:len(disk_usage)-5]
```

## 平稳性测试函数

```bash
from statsmodels.tsa.stattools import adfuller as ADF
diff = 0 
adf = ADF(data['VALUE_C'])

#adf[1]为p值，p值小于0.05认为是平稳的
while adf[1] >= 0.05:
    diff = diff + 1
    adf = ADF(data['VALUE_C'].diff(diff).dropna())
    
print('C盘已使用存储空间的时间序列经过%s阶差分后归于平稳，p值为%s' % (diff,adf[1]))
```


```bash
diff = 0 
adf = ADF(data['VALUE_D'])

#adf[1]为p值，p值小于0.05认为是平稳的
while adf[1] >= 0.05:
    diff = diff + 1
    adf = ADF(data['VALUE_D'].diff(diff).dropna())
    
print('D盘已使用存储空间的时间序列经过%s阶差分后归于平稳，p值为%s' % (diff,adf[1]))
```


## 白噪声检验
#### LB统计量

```bash
from statsmodels.stats.diagnostic import acorr_ljungbox

[[lb],[p]] = acorr_ljungbox(data['VALUE_C'], lags=1)
if p < 0.05:
    print('C盘已使用存储空间的时间序列为 非白噪声序列，对应的p值为：%s'%p)
else:
    print('C盘已使用存储空间的时间序列为 白噪声序列，对应的p值为：%s'%p)
    
[[lb],[p]] = acorr_ljungbox(data['VALUE_C'].diff(1).dropna(),lags=1)
if p < 0.05:
    print('其一阶差分序列为 非白噪声序列，对应的p值为：%s'%p)
else:
    print('其一阶差分为 白噪声序列，对应的p值为：%s'%p)

[[lb],[p]] = acorr_ljungbox(data['VALUE_D'], lags=1)
if p < 0.05:
    print('D盘已使用存储空间的时间序列为 非白噪声序列，对应的p值为：%s'%p)
else:
    print('D盘已使用存储空间的时间序列为 白噪声序列，对应的p值为：%s'%p)
    
[[lb],[p]] = acorr_ljungbox(data['VALUE_D'].diff(1).dropna(),lags=1)
if p < 0.05:
    print('其一阶差分序列为 非白噪声序列，对应的p值为：%s'%p)
else:
    print('其一阶差分为 白噪声序列，对应的p值为：%s'%p)
```

# 2. 时间序列建模

传统的时间序列建模采用ARIMA模型，即Auto Regressive Integrated Moving Average模型。这是一个包括多个子模型的家族，如下

自回归模型（AR）：用变量自身的历史时间数据对变量进行回归，从而预测变量未来的时间数据。
移动平均模型（MA）：移动平均模型关注的是误差项的累加，能够有效消除预测中的随机波动。
自回归移动平均模型（ARMA）：多阶数的AR和MA模型
自回归差分移动平均模型（ARIMA）：带差分计算的ARMA image.png
我们将用ARIMA模型在本例中基于时间序列进行预测；基本步骤如下

对序列绘图，进行 ADF 检验，观察序列是否平稳；
对于非平稳时间序列要先进行 d 阶差分，转化为平稳时间序列
对平稳时间序列分别求得其自相关系数（ACF）和偏自相关系数（PACF），通过对自相关图和偏自相关图的分析，得到最佳的阶数p/q；
基于模型参数d/q/p ，得到ARIMA 模型，最后进行模型检验。


## 获得模型参数 - 最佳p/q值

```bash
def arima_para(time_series, d=1):
    """ 对于给定的平稳时间序列，基于BIC指标的ARIMA模型最佳p/q值
    
    d - 差分阶数；默认为1阶
    """
    
    #定阶 
    pmax = int(len(time_series)/10)   #一般阶数不超过length/10
    qmax = int(len(time_series)/10)
    
    bic_matrix = []         #bic矩阵
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(time_series,(p,d,q)).fit().bic)
            except:
                tmp.append(None)
            
        bic_matrix.append(tmp)
    
    bic_matrix = pd.DataFrame(bic_matrix) 
    p,q = bic_matrix.stack().astype('float64').idxmin()
    
    return p, q

warnings.filterwarnings("ignore")

x_C = data['VALUE_C']
p_C,q_C  = arima_para(x_C)
print('C盘已使用存储空间的时间序列： ARIMA模型最佳p值和q值为:%s、%s'%(p_C,q_C))

x_D = data['VALUE_D']
p_D,q_D  = arima_para(x_D)
print('D盘已使用存储空间的时间序列： ARIMA模型最佳p值和q值为:%s、%s'%(p_D,q_D))
```

## 计算模型预测输出

```bash
arima_C = ARIMA(x_C,(p_C,1,q_C)).fit()
x_pred = arima_C.predict(typ='levels')
```

## 获得原始时间序列的残差

```bash
x_err = (x_pred - x_C).dropna()
```

## 检查残差序列是否是白噪声

```bash
lagnum = 12
lb, p = acorr_ljungbox(x_err, lags = lagnum)
```

## p值小于0.05，认为是非白噪声

```bash
h = (p < 0.05).sum()
if h > 0:
    print('C盘已使用存储空间的时间序列: 模型ARIMA（%s,1,%s）不符合白噪声检验'%(p,q))
else:
    print('C盘已使用存储空间的时间序列: 模型ARIMA（%s,1,%s）符合白噪声检验'%(p,q))
```

## 计算模型预测输出

```bash
arima_D = ARIMA(x_D,(p_D,1,q_D)).fit()
x_pred = arima_D.predict(typ='levels')
```

## 获得原始时间序列的残差

```bash
x_err = (x_pred - x_D).dropna()
```

## 检查残差序列是否是白噪声

```bash
lagnum = 10
lb, p = acorr_ljungbox(x_err, lags = lagnum)
```

## p值小于0.05，认为是非白噪声

```bash
h = (p < 0.05).sum()
if h > 0:
    print('D盘已使用存储空间的时间序列: 模型ARIMA（%s,1,%s）不符合白噪声检验'%(p,q))
else:
    print('D盘已使用存储空间的时间序列: 模型ARIMA（%s,1,%s）符合白噪声检验'%(p,q))
```

# 3. 模型预测

基于学得的ARIMA模型预测最后5个数字，并与原始数据进行比较，评估模型性能

## C盘使用情况预测

```bash
y_forecast = arima_C.forecast(5)[0]
y_true = disk_usage.iloc[len(disk_usage)-5:]['VALUE_C']

comp_C = pd.DataFrame({"C盘使用预测值":y_forecast, "C盘使用实际值":y_true})
comp_C = comp_C.applymap(lambda x :'%.2f'%x)
comp_C
```

## 性能评估

```bash
abs_ = (y_forecast - y_true).abs()
mae_ = abs_.mean()
rmse_ = ((abs_**2).mean())**0.05
mape_ = (abs_/y_true).mean()

print('C盘已使用存储空间的时间序列：\n\n平均绝对误差为：%0.4f,\n均方根误差为：%0.4f,\n平均绝对百分误差为：%0.6f' % (mae_, rmse_, mape_))
```

C盘已使用存储空间的时间序列：

平均绝对误差为：702320.1312,
均方根误差为：3.9350,
平均绝对百分误差为：0.020243


## D盘使用情况预测

```bash
y_forecast = arima_D.forecast(5)[0]
y_true = disk_usage.iloc[len(disk_usage)-5:]['VALUE_D']

comp_D = pd.DataFrame({"D盘使用预测值":y_forecast, "D盘使用实际值":y_true})
comp_D = comp_D.applymap(lambda x :'%.2f'%x)
comp_D
322
```

## 性能评估

```bash
abs_ = (y_forecast - y_true).abs()
mae_ = abs_.mean()
rmse_ = ((abs_**2).mean())**0.05
mape_ = (abs_/y_true).mean()

print('D盘已使用存储空间的时间序列：\n\n平均绝对误差为：%0.4f,\n均方根误差为：%0.4f,\n平均绝对百分误差为：%0.6f' % (mae_, rmse_, mape_))
```

D盘已使用存储空间的时间序列：

平均绝对误差为：1106777.7773,
均方根误差为：4.0449,
平均绝对百分误差为：0.012610

# 总结

看起来模型的性能还不错，因为均方误差和百分比误差都比较低；这也是传统的时间序列分析方法的优势，计算量相对较小速度很快。但是，如果仔细对比预测和实际值，模型的表现还是有待提高。

