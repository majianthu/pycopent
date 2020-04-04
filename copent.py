##################################################################################
###  Estimating Copula Entropy v2.0
###  2019-07-03
###  by MA Jian (Email: majian03@gmail.com)
###
###  Parameters
###    x    	: N * d data, N samples, d dimensions
###    k    	: kth nearest neighbour, parameter for kNN entropy estimation 
###    dtype	: distance type [1: 'Euclidean', others: 'Maximum distance']
###
###  References
###  [1] Ma Jian, Sun Zengqi. Mutual information is copula entropy. 
###      arXiv:0808.0845, 2008.
###  [2] Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. 
###      Physical review E, 2004, 69(6): 066138.
##################################################################################

from scipy.special import digamma
from scipy.stats import rankdata as rank 
from math import gamma, log, pi
from numpy import array, ndarray, abs, sum, sqrt, square

##### calculating distance matrix
def dist(x, dtype = 1):
	(N,d) = x.shape
	xd = ndarray(shape = (N,N), dtype = float)
	
	for i in range(0,N):
		for j in range(i,N):
			if dtype == 1:
				xd[i,j] = sqrt(sum(square(x[i,:]-x[j,:])))
				xd[j,i] = xd[i,j]
			else:
				xd[i,j] = max(abs(x[i,:]-x[j,:]))
				xd[j,i] = xd[i,j]
	
	return xd

##### constructing empirical copula density [1]
def construct_empirical_copula(x):
	(N,d) = x.shape	
	xc = x 
	for i in range(0,d):
		xc[:,i] = rank(x[:,i]) / N
	
	return xc

##### Estimating entropy with kNN method [2]
def entknn(x, k = 3, dtype = 1):
	(N,d) = x.shape
	
	g1 = digamma(N) - digamma(k)
	
	if dtype == 1:	# euciledean distance
		cd = pi**(d/2) / 2**d / gamma(1+d/2)	
	else:	# maximum distance
		cd = 1;
	
	distx = dist(x, dtype)		
	logd = 0
	for i in range(0,N):
		distx[i,:].sort()
		logd = logd + log( 2 * distx[i,k] ) * d / N

	return (g1 + log(cd) + logd)

##### 2-step Nonparametric estimation of copula entropy [1]
def copent(x, k = 3, dtype = 1):
	xarray = array(x)
	xc = construct_empirical_copula(xarray)
	return -entknn(xc, k, dtype)
