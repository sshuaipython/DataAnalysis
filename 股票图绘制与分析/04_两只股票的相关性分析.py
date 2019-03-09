import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md
'''
测试协方差，测试两只股票的相似程度
'''

#定义函数，转换日期函数
def dmy2ymd(dmy):
    dmy=str(dmy,encoding="utf-8")
    date=dt.datetime.strptime(dmy,"%d-%m-%Y").date()#把字符串转为时间格式
    s=date.strftime("%Y-%m-%d")#把时间格式转为字符串
    return s

#加载文件
dates,bhp_closing_prices=np.loadtxt(
            'bhp.csv',
           delimiter=',',
           usecols=(1,6),
           unpack=True,
           dtype=('M8[D],f8'),
           converters={1:dmy2ymd}
           )
# print(dates)

vale_closing_prices=np.loadtxt(
            '../DS/data/vale.csv',
           delimiter=',',
           usecols=(6),
           unpack=True,
           )

#计算两只股票收盘价的相关程度(协方差)
#均值
ave_bhp=bhp_closing_prices.mean()
ave_vale=vale_closing_prices.mean()
#离差
dev_bhp=bhp_closing_prices-ave_bhp
dev_vale=vale_closing_prices-ave_vale
#两组样本的协方差
cov_ab=np.mean(dev_bhp*dev_vale)
print(cov_ab)
#两组样本的相关系数
print(cov_ab/(bhp_closing_prices.std()*vale_closing_prices.std()))

#两组样本的相关矩阵
d=np.corrcoef(bhp_closing_prices,vale_closing_prices)
print(d)

#绘制收盘价的折线图
mp.figure('BHP VS VALE',facecolor="lightgray")
mp.title('BHP VS VALE',fontsize=18)
mp.xlabel('Date',fontsize=14)
mp.ylabel('Price',fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")

#设置刻度定位器,x轴需要显示时间信息
ax=mp.gca()
#x轴主刻度为每周一，次刻度为每天
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_major_formatter(md.DateFormatter('%Y %m %d'))
ax.xaxis.set_minor_locator(md.DayLocator())

#把日期数组类型改为md可识别类型
dates=dates.astype(md.datetime.datetime)
mp.plot(dates,bhp_closing_prices,color="dodgerblue",linewidth=3,linestyle=':',label='bhp_closing_price')
mp.plot(dates,vale_closing_prices,color="orangered",linewidth=3,linestyle=':',label='vale_closing_price')

mp.legend()
#自动格式化日期
mp.gcf().autofmt_xdate()
mp.show()












