import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md
'''
使用多项式拟合两只股票(bhp,vale)收盘价的差价函数
'''

#定义函数，转换日期函数
def dmy2ymd(dmy):
    dmy=str(dmy,encoding="utf-8")
    date=dt.datetime.strptime(dmy,"%d-%m-%Y").date()#把字符串转为时间格式
    s=date.strftime("%Y-%m-%d")#把时间格式转为字符串
    return s

#加载文件
dates,bhp_closing_prices=np.loadtxt(
            '../DS/data/bhp.csv',
           delimiter=',',
           usecols=(1,6),
           unpack=True,
           dtype=('M8[D],f8'),
           converters={1:dmy2ymd}
           )
# print(dates)

vale_closing_prices=np.loadtxt(
            'vale.csv',
           delimiter=',',
           usecols=(6),
           unpack=True,
           )

# 1.读取文件，求得bhp与vale的收盘价的差价
diff_prices=bhp_closing_prices-vale_closing_prices

#绘制收盘价的折线图
mp.figure('BHP DIFF VALE',facecolor="lightgray")
mp.title('BHP DIFF VALE',fontsize=18)
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

mp.xlim(dates[0],dates[-1])

# 2.绘制差价的散点图
mp.scatter(dates,diff_prices,color="dodgerblue",s=60,alpha=0.8,label="Diff_points")

# 3.基于多项式拟合，拟合得到一个多项式方程(多项式系数)
days=dates.astype('M8[D]').astype('int32')
P=np.polyfit(days,diff_prices,4)
# 4.绘制多项式方程的曲线
ys=np.polyval(P,days)
mp.plot(dates,ys,color="orangered",linewidth=3,alpha=0.8,label="Diff_fitted_line")

#找出驻点
Q=np.polyder(P)
xs=np.roots(Q)
ys=np.polyval(P,xs)
print(xs,ys)
xs=xs.astype('M8[D]').astype(md.datetime.datetime)
mp.scatter(xs,ys,color="green",s=60,marker="D",zorder=3,label="points")
mp.annotate((xs[0].strftime("%Y-%m-%d"),ys[0]),xycoords='data',xy=(xs[0],ys[0]),textcoords='offset points',xytext=(0,-50),arrowprops=dict(arrowstyle='->',connectionstyle='angle3'))
mp.annotate((xs[1].strftime("%Y-%m-%d"),ys[1]),xycoords='data',xy=(xs[1],ys[1]),textcoords='offset points',xytext=(0,-50),arrowprops=dict(arrowstyle='->',connectionstyle='angle3'))
mp.annotate((xs[2].strftime("%Y-%m-%d"),ys[2]),xycoords='data',xy=(xs[2],ys[2]),textcoords='offset points',xytext=(-100,-40),arrowprops=dict(arrowstyle='->',connectionstyle='angle3'))


mp.legend()
#自动格式化日期
mp.gcf().autofmt_xdate()
mp.show()




