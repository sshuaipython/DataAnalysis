import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

#定义函数，转换日期函数
def dmy2ymd(dmy):
    dmy=str(dmy,encoding="utf-8")
    date=dt.datetime.strptime(dmy,"%d-%m-%Y").date()#把字符串格式转为时间格式
    s=date.strftime("%Y-%m-%d")#把时间格式转为规定字符串格式
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

#绘制收盘价折线图
mp.plot(dates,closing_prices,color="dodgerblue",linewidth=3,linestyle=':',label='closing_price',alpha=0.8)

#整理蜡烛图所需的颜色
#填充色
rise=closing_prices>opening_prices
color=np.array([('white' if x else 'green') for x in rise])
#边框色
ecolor=np.array([('red' if x else 'green') for x in rise])

#绘制k线图的影线
mp.bar(dates,highest_prices-lowest_prices,0.1,lowest_prices,color=ecolor)

#绘制k线图的实体
mp.bar(dates,closing_prices-opening_prices,0.8,opening_prices,edgecolor=ecolor,color=color)

mp.legend()
#自动格式化日期
mp.gcf().autofmt_xdate()
mp.tight_layout()
mp.show()












