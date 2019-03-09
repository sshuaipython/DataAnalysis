import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.utils as su
import sklearn.metrics as sm
import sklearn.tree as st
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data = pd.read_excel("苏州房价数据处理结果3.xlsx")
# print(data.head())
data_x=np.array(data).T
# print(data_x)

x=[]
for row in range(1,len(data_x)):
    encoder=sp.LabelEncoder()
    if row<len(data_x)-1:
        x.append(encoder.fit_transform(data_x[row]))
    else:
        x.append(data_x[-1])
x=np.array(x).T
# print(x)
y=data_x[0]
# print(y)
fn_house=data.columns[1:]
# print(fn_house)

x,y=su.shuffle(x,y,random_state=7)
train_size=int(len(x)*0.75)
train_x,test_x,train_y,test_y=x[:train_size],x[train_size:],y[:train_size],y[train_size:]


model1=se.RandomForestRegressor(max_depth=10,n_estimators=200,min_samples_split=2)
model1.fit(train_x,train_y)
fi_house1=model1.feature_importances_
pred_test_y1=model1.predict(test_x)
print("随机森林得分：",sm.r2_score(test_y,pred_test_y1))

model2=se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth=10),n_estimators=400,random_state=7)
model2.fit(train_x,train_y)
pred_test_y2=model2.predict(test_x)
fi_house2=model2.feature_importances_
print("AdaBoost得分：",sm.r2_score(test_y,pred_test_y2))

model3 = se.GradientBoostingRegressor(n_estimators = 100,learning_rate=0.2)
model3.fit(train_x,train_y)
pred_test_y3=model3.predict(test_x)
fi_house3=model3.feature_importances_
print("GBRT得分：",sm.r2_score(test_y,pred_test_y3))


#可视化特征重要性排序
plt.figure('House',facecolor='lightgray')
plt.subplot(311)
plt.title('HousePrice_RandomForest',fontsize=16)
plt.ylabel('Importance',fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(axis='y',linestyle=":")
sorted_indices=fi_house1.argsort()[::-1]
pos=np.arange(sorted_indices.size)
plt.bar(pos,fi_house1[sorted_indices],facecolor='deepskyblue',edgecolor='steelblue')
plt.xticks(pos,fn_house[sorted_indices],rotation=30)

plt.figure('House',facecolor='lightgray')
plt.subplot(312)
plt.title('HousePrice_AdaBoost',fontsize=16)
plt.ylabel('Importance',fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(axis='y',linestyle=":")
sorted_indices=fi_house2.argsort()[::-1]
pos=np.arange(sorted_indices.size)
plt.bar(pos,fi_house2[sorted_indices],facecolor='deepskyblue',edgecolor='steelblue')
plt.xticks(pos,fn_house[sorted_indices],rotation=30)

plt.figure('House',facecolor='lightgray')
plt.subplot(313)
plt.title('HousePrice_GBRT',fontsize=16)
plt.ylabel('Importance',fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(axis='y',linestyle=":")
sorted_indices=fi_house3.argsort()[::-1]
pos=np.arange(sorted_indices.size)
plt.bar(pos,fi_house3[sorted_indices],facecolor='deepskyblue',edgecolor='steelblue')
plt.xticks(pos,fn_house[sorted_indices],rotation=30)
plt.show()

