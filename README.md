# copent
Copula熵（Copula Entropy）非参数估计算法


#### 介绍
此分享为非参数Copula熵估计算法。

Copula熵是关于多变量统计相关性度量的数学概念，用于度量非线性、非高斯随机变量之间的统计相关程度 [1]。与皮尔逊相关系数（Pearson Correlation Coefficient）不同，Copula熵是非线性的、高阶的、多变量的，具有更广泛的普适性。

Copula熵的应用广泛，包括：

1）结构学习，学习图结构;

2）变量选择（Variable Selection） [2];

3）因果关系发现（Causality Discovery），用于估计传递熵（Transfer Entropy）[3]。

关于Copula熵的更详细介绍，请参见[作者的博文](http://blog.sciencenet.cn/blog-3018268-978326.html)。


#### 函数说明
copent -- 算法主函数，调用其他两个函数;

construct_empirical_copula -- 算法的第一步，估计经验copula（Empirical Copula）;

entknn -- 算法的第二步，从经验copula利用k近邻法估计copula熵。


#### 参考文献
[1] Ma Jian, Sun Zengqi. Mutual information is copula entropy. Tsinghua Science & Technology, 2011, 16(1): 51-54. See also arXiv preprint, arXiv:0808.0845, 2008.

[2] Ma Jian. Variable Selection with Copula Entropy. arXiv preprint arXiv:1910.12389, 2019.

[3] Ma Jian. Estimating Transfer Entropy via Copula Entropy. arXiv preprint arXiv:1910.04375, 2019.
