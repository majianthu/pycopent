from copent import tst
from numpy import zeros
from numpy.random import multivariate_normal as mnorm

m0 = [0,0]
rho1 = 0.5
v0 = [[1,rho1],[rho1,1]]
s0 = mnorm(m0, v0, 500) # bivariate gaussian 
stat1 = zeros(9)
for i in range(0,9):
	m1 = [i,i]
	s1 = mnorm(m1,v0,500)
	stat1[i] = tst(s0,s1)
	print(stat1[i])

