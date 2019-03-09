import requests
from lxml import etree
import time
import pymongo
import pymysql
import warnings
import csv

class LianjiaSpider:
    def __init__(self):
        self.baseurl='https://su.lianjia.com/ershoufang/pg'
        self.headers={"User-Agent":"Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)"}
        # mongodb
        self.conn = pymongo.MongoClient("192.168.71.128", 27017)
        self.db = self.conn["Lianjia"]
        self.myset = self.db["houseInfo"]
        #mysql
        self.db2 = pymysql.connect("localhost", "root", "123456", "spiderdb", charset="utf8")
        self.cursor = self.db2.cursor()
    
    #获取页面
    def getPage(self,url):
        res=requests.get(url,headers=self.headers)
        res.encoding="utf-8"
        html=res.text
        parseHtml = etree.HTML(html)
        tlist = parseHtml.xpath('//div[@class="info clear"]//div[@class="title"]//a//@href')
        # print(tlist)
        for tlink in tlist:
            self.parsePage(tlink)
    
    #解析页面
    def parsePage(self,tlink):
        res = requests.get(tlink, headers=self.headers)
        res.encoding = "utf-8"
        html = res.text
        parseHtml = etree.HTML(html)
        tprice=parseHtml.xpath('//div[@class="price "]//span[@class="total"]//text()')[0]
        tunit=parseHtml.xpath('//div[@class="price "]//span[@class="unit"]//text()')[0]
        uprice=parseHtml.xpath('//div[@class="price "]//div[@class="unitPrice"]//text()')[0]
        uunit=parseHtml.xpath('//div[@class="price "]//div[@class="unitPrice"]//text()')[1]
        room=parseHtml.xpath('//div[@class="room"]//text()')[0]
        louceng=parseHtml.xpath('//div[@class="room"]//text()')[1].split('/')[0]
        direction=parseHtml.xpath('//div[@class="houseInfo"]//div[@class="type"]//text()')[0]
        area=parseHtml.xpath('//div[@class="houseInfo"]//div[@class="area"]//text()')[0]
        year = parseHtml.xpath('//div[@class="houseInfo"]//div[@class="area"]//text()')[1].split('/')[0]
        communityname=parseHtml.xpath('//div[@class="communityName"]//a[1]//text()')[0]
        areaName=parseHtml.xpath('//div[@class="areaName"]//span[@class="info"]//a[1]//text()')[0]
        areaName2 = parseHtml.xpath('//div[@class="areaName"]//span[@class="info"]//a[2]//text()')[0]
        r = [communityname.strip(),tprice.strip(),tunit.strip(),uprice.strip(),uunit.strip(),room.strip(),louceng.strip(),direction.strip(),area.strip(),year.strip(),areaName.strip(),areaName2.strip()]
        #保存csv文件
        self.savecsv(r)
        #保存到mongodb
        self.savemongo(r)
        # 保存到mysql
        self.savemysql(r)

    #写入csv文件
    def savecsv(self,r):
        with open("house.csv", "a", newline="", encoding="gb18030") as f:
            # 创建写入对象
            writer = csv.writer(f)
            # 调用witerow()方法
            writer.writerow(r)

    # 存MongoDB数据库
    def savemongo(self,r):
        d = {
            "小区名称": r[0],
            "总价": r[1],
            "总价单位": r[2],
            "单价": r[3],
            "单价单位": r[4],
            "户型":r[5],
            "楼层": r[6],
            "朝向":r[7],
            "面积": r[8],
            "年份":r[9],
            "区域": r[10],
            "地点": r[11],
        }
        self.myset.insert_one(d)

    # 存Mysql数据库
    def savemysql(self,r):
        warnings.filterwarnings("ignore")
        ins = 'insert into house(communityname,tprice,tunit,uprice,uunit,room,louceng,direction,area,byear,areaName,areaName2) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        self.cursor.execute(ins, r)
        self.db2.commit()
            
    #主函数
    def workOn(self):
        n=int(input("请输入页数："))
        for pg in range(1,n+1):
            #拼接url
            url=self.baseurl+str(pg)
            self.getPage(url)
            print("第%d页爬取成功"%pg)
            time.sleep(0.1)
    
if __name__=="__main__":
    spider=LianjiaSpider()
    spider.workOn()