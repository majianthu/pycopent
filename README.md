[![PyPI version](https://badge.fury.io/py/copent.svg)](https://pypi.org/project/copent)
# copent
Estimating Copula Entropy and Transfer Entropy

#### Introduction
The nonparametric methods for estimating copula entropy, transfer entropy, and the statistics for multivariate normality test and two-sample test are implemented. The change point detection method based on this two-sample test is also implemented.

The method for estimating copula entropy composes of two simple steps: estimating empirical copula by rank statistic and estimating copula entropy with the KSG method. Copula Entropy is a mathematical concept for multivariate statistical independence measuring and testing, and proved to be equivalent to mutual information. Different from Pearson Correlation Coefficient, Copula Entropy is defined for non-linear, high-order and multivariate cases, which makes it universally applicable. Estimating copula entropy can be applied to many cases, including but not limited to variable selection and causal discovery (by estimating transfer entropy). Please refer to Ma and Sun (2011) <[doi:10.1016/S1007-0214(11)70008-6](http://www.doi.org/10.1016/S1007-0214(11)70008-6)> for more information.

The nonparametric method for estimating transfer entropy composes of two steps: estimating three copula entropy and calculating transfer entropy from the estimated copula entropies. A function for conditional independence testing is also provided. Please refer to Ma (2019) <[arXiv:1910.04375](https://arxiv.org/abs/1910.04375)> for more information.

The copula entropy based statistics for multivariate normality test and two-sample test are implemented. The change point detection method based on this two-sample test is also implemented. Please refer to Ma (2022) <[arXiv:2206.05956](https://arxiv.org/abs/2206.05956)>, Ma (2023) <[arXiv:2307.07247](https://arxiv.org/abs/2307.07247)>, and Ma (2024) <[arXiv:2403.07892](https://arxiv.org/abs/2403.07892)> for more details. 

#### Functions
* copent -- estimating copula entropy;
* construct_empirical_copula -- the first step of the copent function, which estimates empirical copula for data by rank statistics;
* entknn -- the second step of the copent function, which estimates copula entropy from empirical copula with kNN method;
* ci -- conditional independence testing based on copula entropy 
* transent -- estimating transfer entropy via copula entropy
* mvnt -- estimating the copula entropy-based statistic for multivariate normality test
* tst -- estimating the copula entropy-based statistic for two-sample test
* cpd -- single change point detection
* mcpd -- multiple change point detection

#### Parameters
* x: N * d data, N samples, d dimensions
* k: kth nearest neighbour, parameter for kNN entropy estimation. default = 3
* dtype: distance type, can be 'euclidean' or 'chebychev' (for Maximum Distance)
* lag: time lag. default = 1
* s0,s1: two samples with same dimension
* n: repeat time of estimation
* thd	: threshold for the statistic of two-sample test
* maxp	: maximal number of change points
* minseglen : minimal length of binary segmentation

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
data = mnorm(mean1, cov1, 500)
stat1 = mvnt(data)
```

##### two-sample test
```python
from copent import tst
from numpy import zeros
from numpy.random import multivariate_normal as mnorm
m0 = [0,0]
rho1 = 0.5
v0 = [[1,rho1],[rho1,1]]
s0 = mnorm(m0, v0, 400) # bivariate gaussian 
stat1 = zeros(9)
for i in range(0,9):
	m1 = [i,i]
	s1 = mnorm(m1,v0,500)
	stat1[i] = tst(s0,s1)
	print(stat1[i])
```

##### change point detection
```python
from copent import mcpd
import numpy as np
from numpy.random import multivariate_normal as mnorm

n1 = 50
x1 = np.random.normal(0,1,n1)
x2 = np.random.normal(5,1,n1)
x3 = np.random.normal(10,1,n1)
x4 = np.random.normal(1,1,n1)
x = np.concatenate((x1,x2,x3,x4))
pos,maxstat = mcpd(x,thd =0.2)
print(pos)

x1 = mnorm([0,0],[[1,0],[0,1]],n1)
x2 = mnorm([10,10],[[1,0],[0,1]],n1)
x3 = mnorm([5,5],[[1,0],[0,1]],n1)
x4 = mnorm([1,1],[[1,0],[0,1]],n1)
x = np.concatenate((x1,x2,x3,x4))
pos,maxstat = mcpd(x,thd =0.2)
print(pos)
```

#### References
1. Jian Ma and Zengqi Sun. Mutual information is copula entropy. Tsinghua Science & Technology, 2011, 16(1): 51-54. See also arXiv preprint arXiv:0808.0845, 2008.
2. Jian Ma. Estimating Transfer Entropy via Copula Entropy. arXiv preprint arXiv:1910.04375, 2019.
3. Jian Ma. Multivariate Normality Test with Copula Entropy. arXiv preprint arXiv:2206.05956, 2022.
4. Jian Ma. Two-Sample Test with Copula Entropy. arXiv preprint arXiv:2307.07247, 2023.
5. Jian Ma. Change Point Detection with Copula Entropy based Two-Sample Test. arXiv preprint arXiv:2403.07892, 2024.

