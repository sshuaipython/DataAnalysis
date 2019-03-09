## 基于集成模型(随机森林,AdaBoost, GBDT)的苏州二手房的房价分析与预测

### 开发环境及工具

python3+numpy+pandas+matplotlib+sklearn+jupyternotebook

### 获取数据

1. 数据来源：链家二手房(https://su.lianjia.com/ershoufang/)
2. 数据获取时间：2019-01-31
3. 获取方式：使用requests+xpath进行二级子页面爬虫(苏州二手房_链家.py)
4. 保存数据：存入csv文件；存入mysql数据库；存入mongodb数据库

### 数据处理

1. 查看数据的基本信息，共3000条样本，12个特征

           小区名称     总价 总价单位     单价  单价单位    户型   楼层   朝向        面积      年份    区域  \

      0   世茂运河城尚运苑  368.0    万  26958  元/平米  3室2厅  低楼层  南 北  136.51平米    未知年建    姑苏   
      1     阳光水榭花园  190.0    万  21215  元/平米  2室1厅  中楼层    南   89.56平米  2006年建    吴中   
      2     湖岸名家南区  155.0    万  20946  元/平米  2室2厅  低楼层    南      74平米  2007年建    吴中   
      3       香城颐园  200.0    万  22334  元/平米  2室2厅  低楼层  南 北   89.55平米  2012年建    相城   
      4  雅戈尔YAKR公馆  440.0    万  41084  元/平米  2室2厅  高楼层    南   107.1平米  2008年建  工业园区   
    
        地点  
      0  沧浪新城  
      1    城南  
      2    郭巷  
      3    元和  
      4    玲珑   

   (3000, 12)

2. 查看数据中是否有空值

   ```
   <class 'pandas.core.frame.DataFrame'>
   RangeIndex: 3000 entries, 0 to 2999
   Data columns (total 12 columns):
   小区名称    3000 non-null object
   总价      3000 non-null float64
   总价单位    3000 non-null object
   单价      3000 non-null int64
   单价单位    3000 non-null object
   户型      3000 non-null object
   楼层      3000 non-null object
   朝向      3000 non-null object
   面积      3000 non-null object
   年份      3000 non-null object
   区域      3000 non-null object
   地点      3000 non-null object
   dtypes: float64(1), int64(1), object(10)
   memory usage: 281.3+ KB
   ```

3. 处理异常值

   1. 删除异常的楼层样本
   2. 删除无朝向的样本
   3. 删除未知年的样本
   4. 处理异常值后的样本个数(2726, 12)

4. 删除与项目无关特征

5. 对”面积“该特征进行处理，删除”平米“，仅保留数字

6. 对“年份”该特征进行处理，划分成（'5年以内建造'，'6-10年前建造'，'11-15年前建造'，'16-20年前建造'，'超过20年前建造'）这五个类别

7. 数据标准化处理

   ```python
   #对数值字段进行标准化处理
   house['大小']=house['大小'].astype('float')
   house['总价']= (house['总价'] - house['总价'].mean()) / (house['总价'].std())
   house['大小']= (house['大小'] - house['大小'].mean()) / (house['大小'].std())
   ```

8. 将处理后的数据(2726, 8) 保存成exel格式（苏州房价数据处理结果3.xlsx）

   ```
          总价    户型   楼层   朝向         年份    区域  地点        大小
   0 -0.649888  2室1厅  中楼层    南  11-15年前建造    吴中  城南 -0.623108
   1 -0.839060  2室2厅  低楼层    南  11-15年前建造    吴中  郭巷 -1.003166
   2 -0.595839  2室2厅  低楼层  南 北   6-10年前建造    相城  元和 -0.623352
   3  0.701342  2室2厅  高楼层    南  11-15年前建造  工业园区  玲珑 -0.194688
   4 -1.006613  2室2厅  低楼层  东 西   6-10年前建造    吴中  城南 -0.492433
   ```

### 数据分析

1. 分析房价与区域之间的关系
   1. 计算苏州不同区域二手房的最低价，均价和最高价
   2. 计算苏州二手房的平均值
   3. 可视化结果
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E8%8B%8F%E5%B7%9E%E4%BA%8C%E6%89%8B%E6%88%BF%E5%88%86%E6%9E%90/images/Region%26Unitprice.png)
   
   4. 从上图中可以发现，苏州工业园区的房价最高，其次是高新区，姑苏区等。其中工业园区和高新区的二手房均价超过了苏州均价，并且工业园区的最高价远高于其他区域的最高价。
2. 分析工业园区不同地点与房价的关系
   1. 计算苏州工业园区不同地点二手房的最低价，均价和最高价
   2. 计算苏州工业园区二手房的平均值
   3. 可视化结果
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E8%8B%8F%E5%B7%9E%E4%BA%8C%E6%89%8B%E6%88%BF%E5%88%86%E6%9E%90/images/DistrictPrice.png)
   
   4. 从结果中发现，工业园区的玲珑地区房价最高，其次是白糖，湖西CBD，双湖，东沙湖等，胜浦的二手房房价最低。其中有8个地点的二手房均价超过了园区均价，有11个地点的二手房均价超过了苏州二手房均价。
3. 分析苏州二手房数量与区域的关系
   1. 计算苏州不同区域二手房的数量
   2. 计算不同区域二手房的比例
   3. 可视化结果
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E8%8B%8F%E5%B7%9E%E4%BA%8C%E6%89%8B%E6%88%BF%E5%88%86%E6%9E%90/images/Region%26Num2.png)
   
   4. 从可视化结果可以看出，工业园区的二手房数量最多，占比29%，其次是吴中区，姑苏区等。

### 房价预测

1. 读取数据(苏州房价数据处理结果3.xlsx)

   ```python
   data = pd.read_excel("苏州房价数据处理结果3.xlsx")
   data_x=np.array(data).T
   ```

2. 处理数据

   1. 将数据（户型，楼层，朝向，年份，区域，地点）特征进行标签编码

      ```python
      x=[]
      for row in range(1,len(data_x)):
          encoder=sp.LabelEncoder()
          if row<len(data_x)-1:
              x.append(encoder.fit_transform(data_x[row]))
          else:
              x.append(data_x[-1])
      x=np.array(x).T
      y=data_x[0]
      ```

   2. 取总价特征为输出集，其余特征为输入集

   3. 取75%的数据为训练集，25%的数据为测试集

      ```python
      x,y=su.shuffle(x,y,random_state=7)
      train_size=int(len(x)*0.75)
      train_x,test_x,train_y,test_y=x[:train_size],x[train_size:],y[:train_size],y[train_size:]
      ```

3. 建立三种集成模型

   1. 随机森林回归器

      ```python
      model1=se.RandomForestRegressor(max_depth=10,n_estimators=200,min_samples_split=2)
      ```

   2. Adaboost回归器

      ```python
      model2=se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth=10),n_estimators=400,random_state=7)
      ```

   3. GBDT回归器

      ```python
      model3 = se.GradientBoostingRegressor(n_estimators = 100,learning_rate=0.2)
      ```

4. 评估模型

   1. 分别用上面三个模型对训练集进行训练，用测试集进行测试
   2. 得到的得分如下
   
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E8%8B%8F%E5%B7%9E%E4%BA%8C%E6%89%8B%E6%88%BF%E5%88%86%E6%9E%90/images/score.jpg)
   
   3. 三种集成模型的特征重要性
   ![](https://github.com/silencesong/DataAnalysis/blob/master/%E8%8B%8F%E5%B7%9E%E4%BA%8C%E6%89%8B%E6%88%BF%E5%88%86%E6%9E%90/images/feature.png)
   
   4. 房子的大小是影响房子总价的主要特征，其次是地点和区域，符合理论分析

5. 优化模型

   1. 上述的三种模型参数是调参过后的模型参数，得分都差不多，都超过0.85

   2. 为了更好的优化模型，今后还可以从下面几点进行优化

      1. 增加样本特征：由于影响房价的因素很多，本数据集中的特征还需增加，如周围是否有学校、超市、医院等环境因素

      2. 增加样本数量：链家二手房网站中的二手房数量总共只有3000个样本，还需增加样本数量





