library(MASS)

Neg_ll = function(b, x, y, m)
{
  xb = x %*% b
  #used simplified version of negative log likelihood
  Neg_ll = sum(m*log(1+exp(- xb))) + sum((m-y)*xb)
  return (Neg_ll)
} 


Gradient_Cal = function(b, x, y, m)
{
  #saving w as a vector instead of diag matrix 
  w = 1/(1+exp(- (x %*% b))) 
  #use m*w here to save computation time
  Gradient = t(x) %*% (m*w - y) 
  return (Gradient)
}


Sgd_Ada_Grad = function(x, y, m, step=1, lambda=0.01, iter_max=1e5, tol=1e-5)
{
  N = nrow(x)
  P = ncol(x)
  
  #initialize all the variables
  beta = matrix(0, P, iter_max) 
  nll_ini = Neg_ll(rep(0,P), x, y,  m)
  nll = rep(nll_ini, iter_max)
  nll_avg = rep(nll_ini, iter_max)
  nll_ex_avg = rep(nll_ini, iter_max)
  iter = 1
  delta = 1
  G = rep(0, P)
  
  while  ( (iter < iter_max) )
  { 
    #sample 1 data point from the whole data set
    index = sample(1:N, 1)
    xi = x[index,,drop=F]
    yi = y[index]
    mi = m[index,]
    g = Gradient_Cal(beta[,iter], xi, yi, mi)
    G = G + g^2
    #update beta
    beta[,iter+1] = beta[,iter] - step*g/(sqrt(G) + 1E-10)
    #update delta
    delta = abs(max(beta[,iter+1] - beta[,iter]))
    
    #calculate neg ll for whole data set
    nll[iter+1] = Neg_ll(beta[,iter+1], x, y,  m) 
    #calculate neg ll for single data sacle to N
    nll_xi = Neg_ll(beta[,iter+1], xi, yi,  mi)*N 
    #calculate running average neg ll
    nll_avg[iter+1] = (nll_xi + iter*nll_avg[iter])/(iter+1)
    #calculate exp weighted average neg ll with parameter lambda
    nll_ex_avg[iter+1] = lambda*nll_xi + (1-lambda)*nll_ex_avg[iter]
    #keep track of iteration
    iter = iter + 1 
    #nll[1:iter]
  } 
  return (list(iter, beta[, 1:iter], nll[1:iter], nll_avg[1:iter], nll_ex_avg[1:iter]))
}