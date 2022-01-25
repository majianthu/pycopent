##################################################################################
###  Estimating Copula Entropy and Transfer Entropy 
###  2022-01-25
###  by Ma Jian (Email: majian03@gmail.com)
###
###  Parameters
###	x    	: N * d data, N samples, d dimensions
###	k    	: kth nearest neighbour, parameter for kNN entropy estimation. default = 3
###	dtype	: distance type [1: 'Euclidean', 2/others (default): 'Maximum distance']
###	mode	: running mode, 1(default): for speed/small data, 2: for space/large data
###	lag	: time lag. default = 1
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
from numpy import array, ndarray, abs, max, sum, sqrt, square, vstack, zeros
from numpy.random import normal as rnorm

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
##### calculating the distance between two samples
def dist_ij(xi,xj, dtype = 2):
	if dtype == 1:
		return sqrt(sum(square(xi-xj)))
	else: ## dtype = 2
		return max(abs(xi-xj))

##### constructing empirical copula density [1]
def construct_empirical_copula(x):
	(N,d) = x.shape	
	xc = zeros([N,d]) 
	for i in range(0,d):
		xc[:,i] = rank(x[:,i]) / N
	
	return xc

##### Estimating entropy with kNN method [2]
def entknn(x, k = 3, dtype = 2, mode = 1):
	(N,d) = x.shape
	
	g1 = digamma(N) - digamma(k)
	
	if dtype == 1:	# euciledean distance
		cd = pi**(d/2) / 2**d / gamma(1+d/2)	
	else:	# maximum distance
		cd = 1;
	
	logd = 0
	if mode == 1: # for speed / small data
		distx = dist(x, dtype)		
		for i in range(0,N):
			distx[i,:].sort()
			logd = logd + log( 2 * distx[i,k] ) * d / N
	else: # 2, for space / large data
		for i in range(0,N):
			disti = []
			for j in range(0,N):
				disti.append( dist_ij(x[i,:],x[j,:],dtype) )
			disti.sort()
			logd = logd + log( 2 * disti[k] ) * d / N

	return (g1 + log(cd) + logd)

##### 2-step Nonparametric estimation of copula entropy [1]
def copent(x, k = 3, dtype = 2, mode = 1, log0 = False):
	xarray = array(x)

	if log0:
		(N,d) = xarray.shape
		max1 = max(abs(xarray), axis = 0)
		for i in range(0,d):
			if max1[i] == 0:
				xarray[:,i] = rnorm(0,1,N)
			else:
				xarray[:,i] = xarray[:,i] + rnorm(0,1,N) * max1[i] * 0.000005

	xc = construct_empirical_copula(xarray)

	try:
		return -entknn(xc, k, dtype, mode)
	except ValueError: # log0 error
		return copent(x, k, dtype, mode, log0 = True)


##### conditional independence test [3]
##### to test independence of (x,y) conditioned on z
def ci(x, y, z, k = 3, dtype = 2, mode = 1):
	xyz = vstack((x,y,z)).T
	yz = vstack((y,z)).T
	xz = vstack((x,z)).T
	return copent(xyz,k,dtype,mode) - copent(yz,k,dtype,mode) - copent(xz,k,dtype,mode)

##### estimating transfer entropy from y to x with lag [3]
def transent(x, y, lag = 1, k = 3, dtype = 2, mode = 1):
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
	return ci(x2,y,x1,k,dtype,mode)
