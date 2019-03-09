'''
绘制布林带
'''
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

#定义函数，转换日期函数
def dmy2ymd(dmy):
    dmy=str(dmy,encoding="utf-8")
    date=dt.datetime.strptime(dmy,"%d-%m-%Y").date()#把字符串转为时间格式
    s=date.strftime("%Y-%m-%d")#把时间格式转为字符串
    return s

#加载文件
dates,opening_prices,highest_prices,lowest_prices,closing_prices=np.loadtxt(
            'aapl.csv',
           delimiter=',',
           usecols=(1,3,4,5,6),
           unpack=True,
           dtype=('M8[D],f8,f8,f8,f8'),
           converters={1:dmy2ymd}
           )
# print(dates)

#绘制收盘价的折线图
mp.figure('AAPL',facecolor="lightgray")
mp.title('AAPL',fontsize=18)
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
mp.plot(dates,closing_prices,color="dodgerblue",linewidth=3,linestyle=':',label='closing_price',alpha=0.5)

#使用加权卷积绘制五日均线
weights=np.exp(np.linspace(-1,0,5))
core=weights/weights.sum()
# print(core)
sma53=np.convolve(closing_prices,core,'valid')
mp.plot(dates[4:],sma53,color="violet",linewidth=2,label="SMA-5-3",alpha=0.8)

#绘制五日均线的布林带
stds=np.zeros(sma53.size)
for i in range(stds.size):
    stds[i]=closing_prices[i:i+5].std()
#底部支撑线和顶部压力线
lowers=sma53-2*stds
uppers=sma53+2*stds
mp.plot(dates[4:],uppers,color="red",linewidth=2,label="Upper")
mp.plot(dates[4:],lowers,color="green",linewidth=2,label="Lower")
mp.fill_between(dates[4:],lowers,uppers,uppers>lowers,color="dodgerblue",alpha=0.3)


mp.legend()
#自动格式化日期
mp.gcf().autofmt_xdate()
mp.show()




