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
## 主要内容
1. 无监督分箱（*Unsupervised Binning*）
2. 互信息熵计算（*Mutual Infomation Entropy*）
3. 时滞相关性分析（*Time-delayed Correlation Analysis*）

***
## 无监督分箱
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
