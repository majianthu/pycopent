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