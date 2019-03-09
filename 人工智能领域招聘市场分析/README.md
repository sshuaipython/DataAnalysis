# 人工智能领域的招聘市场数据分析

### 环境及工具

python3+pandas+numpy+matplotlib+pyecharts+jieba+jupyter notebook

### 过程

#### 1.读取数据，清洗数据

读取数据（51job/人工智能.csv），去除空行后数据从14383个样本变成14304个

#### 2.词频统计，汇出词云

1. 使用jieba库对工作需求这一特征的内容进行中文分词(过滤标点及特殊符号，并过滤掉停用词汇)
![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/1jieba.png)

2. 根据人工智能行业现状自行编辑的词库(keyword.txt)，进行词频统计
![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/2cipin.png)

3. 由于jieba库中的词太具有广泛性，自编词库统计的结果比jieba库统计的更客观一些

#### 3.全国城市的岗位需求量分析

1. 统计出人工智能岗位需求量前十名的城市
2. 使用柱状图进行可视化
![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/citypos.png)

3. 从图中可看出，人工智能相关职位上海需求量最高，之后分别是北京、深圳、广州、杭州、成都等城市

#### 4.长三角两城市(上海和杭州)工作岗位的分布分析

1. 统计上海地区工作岗位的分布
   1. 数据可视化，热力图
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/shanghai1.png)
   
   2. 数据可视化，饼图
   
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/shanghai2.png)
   
   3. 上海市的人工智能相关职位需求量浦东新区最高，随后是徐汇区、嘉定区、杨浦区、静安区等
2. 统计杭州地区工作岗位的分布
   1. 数据可视化，热力图
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/hangzhou1.png)
   
   2. 数据可视化，饼图
   
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/hangzhou2.png)
   
   3. 杭州地区的人工智能相关职位需求量余杭区最高，之后分别是滨江区、西湖区、江干区等

#### 5.人工智能行业对学历要求的分析

1. 51job网站数据
   1. 读取并整理数据(51job/51job.csv)，共11741个样本，12个特征
   2. 统计并可视化学历要求图
   
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/edu51job.png)
   
   3. 51job网站的数据中，前四位的是本科，大专，无要求，硕士。
2. 拉勾网数据
   1. 读取并整理数据(51job/51job.csv)，共4733个样本，13个特征4733,
   2. 统计并可视化学历要求图
   
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/edu_lagou.png)
   
   3. 拉勾网站的数据中，学历要求占比排序分别是本科，硕士，大专，学历不限，博士
   
3. 根据人工智能行业现状及当前的发展趋势，拉勾网的数据较51job的数据结果更可靠谱

#### 6.人工智能行业相关职位薪资对比

1. 读取并整理数据(拉勾文件夹中的所有文件)
2. 计算出各职位较低薪资均值，较高薪资均值，以及整体均值
3. 数据可视化，柱状图
![](https://github.com/silencesong/DataAnalysis/blob/master/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E9%A2%86%E5%9F%9F%E6%8B%9B%E8%81%98%E5%B8%82%E5%9C%BA%E5%88%86%E6%9E%90/Images/salary.png)



### 结论

1. 随着科技的发展，人工智能行业如雨后春笋般迅速崛起，目前全国主要一线城市都提供了很多相关的职位。上海、北京、深圳、广州、杭州、成都等城市的人工智能行业发展迅猛，也提供了很多就业机会。
2. 分析了长三角两个城市，上海和杭州的职位分布情况，上海地区的分布主要集中在浦东新区、徐汇区、嘉定区、杨浦区、静安区；杭州地区的分布集中在余杭区、滨江区、西湖区、江干区。
3. 人工智能行业目前对学历的要求以本科和硕士学历为主，还有一些对博士的需求
4. 从人工智能行业的薪资图中看出，整体薪资分布很高，月薪均值在24055左右。在该领域，NLP方向薪资最高，图像识别方向薪资最低。

