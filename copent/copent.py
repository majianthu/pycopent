##################################################################################
###  Estimating Copula Entropy and Transfer Entropy 
###  2021-03-20
###  by Ma Jian (Email: majian03@gmail.com)
###
###  Parameters
###	x    	: N * d data, N samples, d dimensions
###	k    	: kth nearest neighbour, parameter for kNN entropy estimation 
###	dtype	: distance type [1: 'Euclidean', others: 'Maximum distance']
### lag		: time lag
###
###  References
###  [1] Ma Jian, Sun Zengqi. Mutual information is copula entropy. 
###      arXiv:0808.0845, 2008.
###  [2] Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. 
###      Physical review E, 2004, 69(6): 066138.
###  [3] Ma, Jian. Estimating Transfer Entropy via Copula Entropy. 
###      arXiv preprint arXiv:1910.04375 (2019).
##################################################################################

from scipy.special import digamma
from scipy.stats import rankdata as rank 
from math import gamma, log, pi
from numpy import array, ndarray, abs, sum, sqrt, square, vstack

##### calculating distance matrix
def dist(x, dtype = 2):
	(N,d) = x.shape
	xd = ndarray(shape = (N,N), dtype = float)
	
	for i in range(0,N):
		for j in range(i,N):
			if dtype == 1:
				xd[i,j] = sqrt(sum(square(x[i,:]-x[j,:])))
				xd[j,i] = xd[i,j]
			else: ## dtype = 2
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
def entknn(x, k = 3, dtype = 2):
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
def copent(x, k = 3, dtype = 2):
	xarray = array(x)
	xc = construct_empirical_copula(xarray)
	return -entknn(xc, k, dtype)

##### conditional independence test [3]
##### to test independence of (x,y) conditioned on z
def ci(x, y, z, k = 3, dtype = 2):
	xyz = vstack((x,y,z)).T
	yz = vstack((y,z)).T
	xz = vstack((x,z)).T
	return copent(xyz,k,dtype) - copent(yz,k,dtype) - copent(xz,k,dtype)

##### estimating transfer entropy from y to x with lag [3]
def transent(x, y, lag = 1, k = 3, dtype = 2):
	xlen = len(x)
	ylen = len(y)
	if (xlen > ylen):
		l = ylen
	else:
		l = xlen
	if (l < (lag + k + 1)):
		return 0
	x1 = x[0:(l-lag)]
	x2 = x[lag:l]
	y = y[0:(l-lag)]
	return ci(x2,y,x1,k,dtype)
