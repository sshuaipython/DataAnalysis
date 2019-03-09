import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md

# 解决中文字乱码
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 数据读取
data = pd.read_csv('QIY.csv', engine = 'python')
# print(data.head(10))
# print(data.shape)

# 数据清洗 - 去除空值
# 文本型字段空值改为“缺失数据”，数字型字段空值改为 0
def data_cleaning(df):
    cols = df.columns
    for col in cols:
        if df[col].dtype ==  'object':
            df[col].fillna('缺失数据', inplace = True)# 该函数可以将任意数据内空值替换
        else:
            df[col].fillna(0, inplace = True)
    return(df)
data_c1 = data_cleaning(data)
# print(data_c1.head(10))

# 数据清洗 - 时间标签转化,将时间字段改为时间标签
def data_time(df,*cols):
    for col in cols:
        df[col] = df[col].str.replace('年','.')
        df[col] = df[col].str.replace('月','.')
        df[col] = df[col].str.replace('日','')
        df[col] = pd.to_datetime(df[col])# 该函数将输入列名的列，改为DatetimeIndex格式
    return df
data_c2 = data_time(data_c1,'数据获取日期')
# print(data_c2.head(10))

# 问题1 分析出不同导演电影的好评率，并筛选出TOP20
# 计算统计不同导演的好评率
df_q1 = data_c2.groupby('导演')[['好评数','评分人数']].sum()
df_q1['好评率'] = df_q1['好评数'] / df_q1['评分人数']
result_q1 = df_q1.sort_values(['好评率'], ascending=False)[:20]
print(result_q1)
fig1 = plt.figure(num=1,figsize=(12,7))
result_q1['好评率'].plot(kind='bar',color = 'dodgerblue', width = 0.8,alpha = 0.8)
plt.ylim(0.978,1)
plt.title('TOP20导演电影的好评率',fontsize=20)
plt.grid(axis='y',linestyle=':')
plt.xlabel('导演',fontsize=16)
plt.ylabel('好评率',fontsize=16)
plt.tick_params(labelsize=10)
# plt.show()

# 问题2 统计分析2001-2016年每年评影人数总量
# 筛选出不同年份的数据，去除‘上映年份’字段缺失数据
q2data1 = data_c2[['导演','上映年份','整理后剧名']].drop_duplicates()
q2data1 = q2data1[q2data1['上映年份'] != 0]
# print(q2data1,q2data1.shape)#(2144,3)

# 求出不同剧的评分人数、好评数总和
q2data2 = data_c2.groupby('整理后剧名').sum()[['评分人数','好评数']]
# print(q2data2,q2data2.shape)#(2597,2)

# 合并数据，得到不同年份，不同剧的评分人数、好评数总和
q2data3 = pd.merge(q2data1,q2data2,left_on='整理后剧名',right_index=True)
# print(q2data3,q2data3.shape)#(2144,5）

# 按照电影上映年份统计，评分人数量
q2data4 = q2data3.groupby('上映年份').sum()[['评分人数','好评数']]

# 创建折线图
fig2 = plt.figure(num=2,figsize=(12,4))
plt.plot(range(2000,2017),q2data4['评分人数'].loc[2000:],color = 'limegreen',alpha = 0.8,linewidth=5)
plt.xlim(2000,2016)
plt.title('2000-2016年每年评影人数总量统计',fontsize=20)
plt.xlabel('上映年份',fontsize=16)
plt.xlabel('评分人数',fontsize=16)
plt.grid(linestyle=":")
plt.tick_params(labelsize=10)
# plt.show()

#统计评分人数YOP20的影视剧
top20 = q2data3.sort_values(['评分人数'], ascending=False)[:20]
# print(top20)
fig3 = plt.figure(num=3)
plt.bar(range(20),top20['评分人数'],width=0.8,color="dodgerblue",tick_label=top20['整理后剧名'])
plt.xticks(rotation=90)
plt.title('评分人数TOP20的影视剧',fontsize=20)
plt.grid(axis='y',linestyle=':')
plt.xlabel('剧名',fontsize=16)
plt.ylabel('评分人数',fontsize=16)
plt.tick_params(labelsize=10)
for i,j in zip(range(20),top20['上映年份']):
    plt.text(i-0.5,0.6e8,'%d'%int(j),color='k',fontsize=9,rotation=45)
# plt.show()

# 创建函数得到外限最大最小值
# 查看异常值
fig,axes = plt.subplots(4,4,figsize=(10,16))
start = 2001
for i in range(4):
    for j in range(4):
        data = q2data3[q2data3['上映年份'] == start]
        #创建矩阵箱型图
        data[['评分人数','好评数']].boxplot(whis = 3, return_type='dict',ax = axes[i,j])
        start += 1
plt.show()
# 发现基本每年的数据中都有异常值，且为极度异常

# 创建函数，得到外限最大最小值
a = q2data3[q2data3['上映年份'] == 2001]
def data_error(df,col):
    q1 = df[col].quantile(q=0.25)  # 上四分位数
    q3 = df[col].quantile(q=0.75)  # 下四分位数
    iqr = q3 - q1   # IQR
    tmax = q3 + 3 * iqr  # 外限最大值
    tmin = q3 - 3 * iqr  # 外限最小值
    return(tmax,tmin)

snum=[]
anum=[]
# 查看异常值信息
for i in range(2001,2017):
    datayear = q2data3[q2data3['上映年份'] == i]  # 筛选该年度的数据
    t = data_error(datayear,'评分人数')  # 得到外限最大最小值
    snum.append(len(datayear))
    print('%i年共有%i条数据' % (i,len(datayear)),end=",")  # 查看每年的数据量
    anum.append(datayear[datayear['评分人数'] > t[0]].shape[0])
    print("异常数据有%s条:"% datayear[datayear['评分人数'] > t[0]].shape[0])
    print(datayear[datayear['评分人数'] > t[0]])  # 查看评分人数大于外限最大值的异常值
    print('-------\n')

snum=np.array(snum)
anum=np.array(anum)
ratio=anum/snum
data=pd.DataFrame(snum,columns=['sum'])
data['abnormal']=anum
data['ratio']=ratio
data.index=list(range(2001,2017))
print(data)
