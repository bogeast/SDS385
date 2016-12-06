library(MASS)
library(Matrix)
library(truncnorm)
library(parallelMCMCcombine)
library(MCMCpack)
library(abind)


Cmc_Probit = function(x, y, s=100, burnin=5000, mcmc=10000, mu0=0, tau0=0)
{
  N = nrow(x)
  folds = cut( seq(1,N), breaks=s, labels=FALSE)
  cmc_sample = NULL

  for(i in 1:s)
  {
    shard_index = which(folds==i,arr.ind=TRUE)
    shardX = x[shard_index, ]
    shardY = y[shard_index ]
    shardbeta = Bayes_Probit(x=shardX, y=shardY, burnin=burnin, mcmc=mcmc, mu0=mu0, tau0=tau0)
    cmc_sample = abind(cmc_sample, shardbeta, along=3)
  }
  cmc_combine = consensusMCcov(cmc_sample, shuff = FALSE)
  return(cmc_combine)
}


Bayes_Probit = function(x, y, burnin=5000, mcmc=10000, mu0=0, tau0=0)
{
  N = nrow(x)
  P = ncol(x)
  
  beta = matrix(0, P, mcmc) 
  Z = rnorm(N)
  
  for (iter in 1:mcmc)
  {
    for ( i in 1:N)
    {
      mu = crossprod(x[i,], beta[,iter])
      if (y[i] == 1) {
        Z[i] = rtruncnorm(1,a=0, b=Inf, mean=mu, sd=1)
      } else {
        Z[i] = rtruncnorm(1,a=-Inf, b=0, mean=mu, sd=1)
      }
    }
    V = solve(tau0 + t(x) %*% x + diag(10-6, P))
    B = V %*% (tau0*mu0 + t(x) %*% Z)
    beta[, iter] = mvrnorm(1, mu=B, Sigma=V) 
  }
  return(beta[, (burnin+1):mcmc])
}