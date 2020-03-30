# copent
Copula Entropy Non-Parametric Estimation Algorithm


#### Introduction
Copula Entropy is a mathematical concept for statistical independence measurement [1]. Different from Pearson Correlation Coefficient, Copula Entropy is defined for non-linear, high-order and multivariate case, which is universally applicable.

It enjoys wide applications, include but not limit to：

1）Structure Learning;

2）Variable Selection [2];

3）Causality Discovery, (Estimating Transfer Entropy) [3].

For more information, please refer to [1-3]. For more information in Chinese, please follow [this link](http://blog.sciencenet.cn/blog-3018268-978326.html).

#### Functions
copent -- main function;

construct_empirical_copula -- the first step of the algorithm, which estimates empirical copula for data by rank statistics;

entknn -- the second step of the algorithm, which estimates copula entropy from empirical copula with kNN method.

#### References
[1] Ma Jian, Sun Zengqi. Mutual information is copula entropy. Tsinghua Science & Technology, 2011, 16(1): 51-54. See also arXiv preprint, arXiv:0808.0845, 2008.

[2] Ma Jian. Variable Selection with Copula Entropy. arXiv preprint arXiv:1910.12389, 2019.

[3] Ma Jian. Estimating Transfer Entropy via Copula Entropy. arXiv preprint arXiv:1910.04375, 2019.
