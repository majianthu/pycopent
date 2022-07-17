import numpy as np
from numpy.random import multivariate_normal as mnorm
from copent import mvnt

mean1 = [0,0]
cov1 = [[1,0.65],[0.65,1]]
data = mnorm(mean1, cov1, 800) # bivariate gaussian 
stat1 = mvnt(data)
str1 = "true value: 0; statistic: %0.3f" % stat1
print(str1)
