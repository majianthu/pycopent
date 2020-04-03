##################################################################################
###  Estimating Copula Entropy from data
###  2019-07-03
###  by MA Jian (Email: majian03@gmail.com)
###
###  Parameters
###   x		: N * d data, N samples, d dimensions
###   k 	: kth nearest neighbour, parameter for kNN entropy estimation 
###   dm	: distance type [1: 'Euclidean', others: 'Maximum distance']
###
###  References
###  [1] Ma Jian, Sun Zengqi. Mutual information is copula entropy. 
###      arXiv:0808.0845, 2008.
###  [2] Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. 
###      Physical review E, 2004, 69(6): 066138.
##################################################################################
copent<-function(x,k=3,dm=2){
  xc = construct_empirical_copula(x)
  -entknn(xc,k,dm)
}

construct_empirical_copula<-function(x){
  require(matlab)
  require(Matrix)
  
  dimx = size(as.matrix(x));
  rx = dimx[1];
  lx = dimx[2];
  
  xrank = x
  for(i in 1:lx){
    xrank[,i] = rank(x[,i])
  }
  
  xrank / rx
}

entknn<-function(x,k=3,dm=2){
  require(Matrix)
  require(matlab)
  
  x = as.matrix(x)
  # get the dimension of data x
  N = dim(x)[1];
  d = dim(x)[2];
  
  g1 = digamma(N) - digamma(k);
  
  if (dm == 1){	# euciledean distance
    cd = pi^(d/2) / 2^d / gamma(1+d/2);	
    distx = as.matrix(dist(x));
  }
  else {	# maximum distance
    cd = 1;
    distx = as.matrix(dist(x,method = "maximum"));
  }
  
  logd = 0;
  for(i in 1:N){
    distx[i,] = sort(distx[i,]);
    logd = logd + log( 2 * distx[i,k+1] ) * d / N;
  }
  
  g1 + log(cd) + logd
}
