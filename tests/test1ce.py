from numpy.random import multivariate_normal as mnorm
from math import log 
import copent

rho = 0.8
mean1 = [0,0]
cov1 = [ [1,rho],[rho,1] ]
data = mnorm(mean1, cov1, 800) # bivariate gaussian 
truevalue1 =  - log(1 - rho**2) / 2  # true value of copula entropy
copent1 = copent.copent(data) # estimated copula entropy
str1 = "true value: %0.3f; Estimated CE: %0.3f" % (truevalue1, copent1)
print(str1)
