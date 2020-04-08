# multivariate-correlation-analysis
多变量相关性分析



#### 一、背景介绍

相关性分析可以用于定量分析变量间的非线性作用关系。

> Measures for multivariate correlation analysis have three important applications: detection, quantification and ranking.

现有的大多数相关性分析方法是成对分析（pairwise analysis），实际情况下变量之间可能存在多元耦合关系；另外的一些算法虽然适用于多元相关性分析，但是分析结果会随着变量维数的变化而变化（biased），所以我们需要一种更适用的算法来处理这种问题。

> If bivariate based measures are used to identify multivariate correlations, through pair- wise aggregation, multivariate correlations can potentially be overlooked.

> We argue that multivariate correlation measures should be *comparable*, *interpretable*, *scalable* and *unbiased*.

文中提出了一种无偏多元相关性分析方法（unbiased multivariate correlation measure, UMC），与其他已有的分析方法在可并行性、可解释性、可扩展性和无偏性上的表现对比如下：

<img src="assets/方法对比.png" alt="方法对比" style="zoom: 33%;" />



#### 二、原理阐述

对于含有$d$维实值变量的数据集$D=\{X_i\}_{i = 1}^{d}$，样本量为$n$。相关性分析$M$本质上是需要区分$D$中各变量实际分布与独立分布的差异程度，即：
$$
M(D)=diff\left(
	p(X_1, ..., X_d), \Pi_{i = 1}^{d}p(X_i)
\right)
$$
若令$diff$为KL散度，上式变为全相关测量（Total Correlation measure, TC measure）：
$$
\begin{align*}
TC(D) &= KL \left(
p(X_1, ..., X_d) || \Pi_{i = 1}^{d}p(X_i)
\right) \\
&= [\Sigma_{i=1}^{d}H(X_i)] - H(X_1, ..., X_d)
\end{align*}
$$
其中，对高维数据的联合分布熵$H(X_1, ..., X_d)$的计算可能会遇到“empty space problem”，为了避免此类问题，将上述KL散度计算公式进行改写：
$$
KL \left(p(X_1, ..., X_d) || \Pi_{i = 1}^{d}p(X_i)\right) = 
KL\left(p(X_2|X_1)||p(X_2)\right) + \cdots + KL\left(p(X_d|X_1, ..., X_{d-1})||p(X_d)\right)
$$
这种改写方式对$X$的顺序是敏感的，为了消除排序的影响，进一步写作：
$$
M(D) \sim \max_{\sigma \in F_{d}} \sum_{i=2}^{d}diff \left(
	p(X_{\sigma(i)}), p(X_{\sigma(i)}|X_{\sigma(1)}, ..., X_{\sigma(i-1)})
\right)
$$
本文中的UMC算法即是基于式（3）而提出的，其中的$diff$函数通过累积熵（cumulative entropy）实现，累积熵的一大优点是无需数据离散化处理，没有数据信息损失。连续随机变量的累积熵与信息熵形式类似，定义如下：
$$
h(Z)=- \int P(Z \leq z) \log P(Z \leq z) \rm d z
$$
其中$P(Z \leq z)$表示变量$Z$的累积分布函数（Culmulative Distribution Function, CDF），由于$0 \leq P(Z \leq z) \leq 1$，所以$h(Z) \geq 0$。同时定义条件累积熵：
$$
h(Z|Y) = \int h(Z|y)p(y) {\rm d}y, y \in {\rm domain}(Y)
$$




#### 参考文献：

1. Y. Wang, etc.: Unbiased Multivariate Correlation Analysis, AAAI-17, 2017.