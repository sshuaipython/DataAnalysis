import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

#读取训练数据
x,y=[],[]
with open('single.txt','r') as f:
    for line in f.readlines():
        data=[float(substr) for substr in line.split(',')]
        x.append(data[:-1])
        y.append(data[-1])
x=np.array(x) #二维数组形式的输入矩阵，一行一样本，一列一特征
y=np.array(y) #一维数组形式的输出序列，每个元素对应一个输入样本
# print(x)
# print(y)

#创建线性回归器
model=lm.LinearRegression()
#用已知的输入和输出来训练线性回归器
model.fit(x,y)
#根据给定的输入来预测对应的输出
pred_y=model.predict(x)

#评估指标
error=sm.mean_absolute_error(y,pred_y)#平均绝对值误差
print(error)#0.54828
error2=sm.mean_squared_error(y,pred_y)#平均平方误差
print(error2)#0.43607
error3=sm.median_absolute_error(y,pred_y)#中位绝对值误差
print(error3)#0.53566
error4=sm.r2_score(y,pred_y)#R2得分[0,1],越大越好
print(error4)#0.73626

#可视化回归曲线
mp.figure('Linear Regression',facecolor='lightgray')
mp.title('Linear Regression',fontsize=20)
mp.xlabel('X',fontsize=14)
mp.ylabel('Y',fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
mp.scatter(x,y,c='dodgerblue',alpha=0.75,s=60,label='Sample')
sorted_indecies=x.T[0].argsort()
mp.plot(x[sorted_indecies],pred_y[sorted_indecies],c='orangered',label='Regression')

mp.legend()
mp.show()
