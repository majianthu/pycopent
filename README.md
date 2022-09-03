[![PyPI version](https://badge.fury.io/py/copent.svg)](https://pypi.org/project/copent)
# copent
Estimating Copula Entropy and Transfer Entropy

#### Introduction
The nonparametric methods for estimating copula entropy, transfer entropy, and the statistic for multivariate normality test are implemented. 

The method for estimating copula entropy composes of two simple steps: estimating empirical copula by rank statistic and estimating copula entropy with the KSG method. Copula Entropy is a mathematical concept for multivariate statistical independence measuring and testing, and proved to be equivalent to mutual information. Different from Pearson Correlation Coefficient, Copula Entropy is defined for non-linear, high-order and multivariate cases, which makes it universally applicable. Estimating copula entropy can be applied to many cases, including but not limited to variable selection and causal discovery (by estimating transfer entropy). Please refer to Ma and Sun (2011) <[doi:10.1016/S1007-0214(11)70008-6](http://www.doi.org/10.1016/S1007-0214(11)70008-6)> for more information.

The nonparametric method for estimating transfer entropy composes of two steps: estimating three copula entropy and calculating transfer entropy from the estimated copula entropies. A function for conditional independence testing is also provided. Please refer to Ma (2019) <[arXiv:1910.04375](https://arxiv.org/abs/1910.04375)> for more information.

The copula entropy based statistic for multivariate normality test is implemented. Please refer to Ma (2022) <[arXiv:2206.05956](https://arxiv.org/abs/2206.05956)> for more details.

#### Functions
* copent -- estimating copula entropy;
* construct_empirical_copula -- the first step of the copent function, which estimates empirical copula for data by rank statistics;
* entknn -- the second step of the copent function, which estimates copula entropy from empirical copula with kNN method;
* ci -- conditional independence testing based on copula entropy 
* transent -- estimating transfer entropy via copula entropy
* mvnt -- the copula entropy-based statistic for multivariate normality test

#### Parameters
* x: N * d data, N samples, d dimensions;
* k: kth nearest neighbour, parameter for kNN entropy estimation. default = 3;
* dtype: distance type, can be 'euclidean' or 'chebychev' (for Maximum Distance);
* lag: time lag. default = 1;
* mode: running mode, 1(default) for speed/small data, 2 for space/large data.

#### Installation
The package can be installed from PyPI directly:
```
pip install copent
```
The package can be installed from Github:
```
pip install git+https://github.com/majianthu/pycopent.git
```
#### Usage Examples
##### estimating copula entropy 
```python
from numpy.random import multivariate_normal as mnorm
import copent
rho = 0.6
mean1 = [0,0]
cov1 = [ [1,rho],[rho,1] ]
x = mnorm(mean1,cov1,200) # bivariate gaussian 
ce1 = copent.copent(x) # estimated copula entropy
```

##### estimating transfer entropy 
```python
from copent import transent
from pandas import read_csv
import numpy as np
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
prsa2010 = read_csv(url)
# index: 5(PM2.5),6(Dew Point),7(Temperature),8(Pressure),10(Cumulative Wind Speed)
data = prsa2010.iloc[2200:2700,[5,8]].values
te = np.zeros(24)
for lag in range(1,25):
	te[lag-1] = transent(data[:,0],data[:,1],lag)
	str = "TE from pressure to PM2.5 at %d hours lag : %f" %(lag,te[lag-1])
	print(str)
```

##### multivariate normality test
```python
from numpy.random import multivariate_normal as mnorm
from copent import mvnt
mean1 = [0,0]
cov1 = [[1,0.65],[0.65,1]]
data = mnorm(mean1, cov1, 500) # bivariate gaussian 
stat1 = mvnt(data)
```

#### References
1. Jian Ma and Zengqi Sun. Mutual information is copula entropy. Tsinghua Science & Technology, 2011, 16(1): 51-54. See also arXiv preprint arXiv:0808.0845, 2008.
2. Jian Ma. Estimating Transfer Entropy via Copula Entropy. arXiv preprint arXiv:1910.04375, 2019.
3. Jian Ma. Multivariate Normality Test with Copula Entropy. arXiv preprint arXiv:2206.05956, 2022.

