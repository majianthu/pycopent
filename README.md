# copent
Estimating Copula Entropy Non-parametrically in Python


#### Introduction
The nonparametric method for estimating copula entropy is implemented. The method composes of two simple steps: estimating empirical copula by rank statistic and estimating copula entropy with k-Nearest-Neighbour method. Copula Entropy is a mathematical concept for multivariate statistical independence measuring and testing, and proved to be equivalent to mutual information. Different from Pearson Correlation Coefficient, Copula Entropy is defined for non-linear, high-order and multivariate cases, which makes it universally applicable. Estimating copula entropy can be applied to many cases, including but not limited to variable selection [2] and causal discovery (by estimating transfer entropy) [3]. Please refer to Ma and Sun (2011) <doi:10.1016/S1007-0214(11)70008-6> for more information. For more information in Chinese, please follow [this link](http://blog.sciencenet.cn/blog-3018268-978326.html).

#### Functions
* copent -- main function;

* construct_empirical_copula -- the first step of the algorithm, which estimates empirical copula for data by rank statistics;

* entknn -- the second step of the algorithm, which estimates copula entropy from empirical copula with kNN method.

#### Installation
The package can be installed from PyPI directly:
```
pip install copent
```
The package can be installed from Github:
```
pip install git+https://github.com/majianthu/copent.git
```
#### Usage Examples
```
#### Example for copent.py
from numpy.random import multivariate_normal as mnorm
import copent
rho = 0.6
mean1 = [0,0]
cov1 = [ [1,rho],[rho,1] ]
x = mnorm(mean1,cov1,200) # bivariate gaussian 
ce1 = copent.copent(x) # estimated copula entropy
```

#### References
1. Ma Jian, Sun Zengqi. Mutual information is copula entropy. Tsinghua Science & Technology, 2011, 16(1): 51-54. See also arXiv preprint, arXiv:0808.0845, 2008.

2. Ma Jian. Variable Selection with Copula Entropy. arXiv preprint arXiv:1910.12389, 2019.

3. Ma Jian. Estimating Transfer Entropy via Copula Entropy. arXiv preprint arXiv:1910.04375, 2019.
