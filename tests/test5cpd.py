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


