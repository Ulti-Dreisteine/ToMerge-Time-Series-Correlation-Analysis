<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


![封面](img/cover_picture.png)

***
## &ensp; 主要内容
1. 无监督分箱（*Unsupervised Binning*）
2. 互信息熵计算（*Mutual Infomation Entropy*）
3. 时滞相关性分析（*Time-delayed Correlation Analysis*）

***
### 1. &ensp; 无监督分箱
数据分箱是一种将多个连续值分组为较少数量的“分箱”的方法，其意义在于：  
* 统计需要
* 减少次要观测误差的影响
* 离散特征的增加和减少都很容易，易于模型的快速迭代
* 特征离散化，降低模型过拟合风险，增加模型鲁棒性

数据分箱按照有无标签可以分为有监督分箱和无监督分箱两类方法，常见的有监督分箱有：
* 卡方分箱
* KS分箱

无监督分箱有：
* 等频分箱
* 等距分箱

在本项目中，我们会选择使用无监督分箱对时间序列样本进行统计。

#### 1.1 &ensp; 一维序列分箱    
我们在*unsupervised_data_binning.series_binnging*中提供了对一维序列进行无监督分箱的方法。首先，初始化一个*SeriesBinning*对象：

```
binning = SeriesBinning(x, x_type)
```

其中，*x*为时间序列, 可以为list也可以是np.ndarray对象，程序将自动将其转换为一维np.ndarray数组：  

```
self.x = np.array(x).flatten()  # flatten处理
```

x_type表示序列x中值的类型，为连续数值型“continuous”或离散类别型“discrete”:

```
if x_type not in VALUE_TYPES_AVAILABLE:
  raise ValueError('Param x_type {} not in value_types_availabel = {}.'.format(x_type, VALUE_TYPES_AVAILABLE))
```  

其中*SeriesBinning*对象初始化后便会获得属性*stat_params*以记录序列x的均值、标准差、中位数、上下四分位数等统计参数，以作后用。  
接下来便可对*x*进行分箱操作了，可以调用如下接口实现：  

```
freq_ns, labels = self.isometric_binning(bins)                    # 等距分箱
freq_ns, labels = self.quasi_chi2_binning(init_bins, final_bins)  # 拟卡方分箱
freq_ns, labels = self.label_binning()                            # 根据类别标签进行分箱
```

也可以统一调用接口实现：  

```
freq_ns, labels = self.series_binning(method, params)             # 通用接口
```  

其中*method*支持“isometric”、“quasi_chi2”和“label”，*params*对应设置如下：

```
if method == 'isometric':
  assert 'bins' in params.keys()
elif method == 'quasi_chi2':
  assert 'init_bins' in params.keys()
  assert 'final_bins' in params.keys()
elif method == 'label':
  pass
else:
  raise ValueError('Unknown method "{}"'.format(method))
```  
