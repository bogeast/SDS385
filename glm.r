library(MASS)

### Data Prep ###
wdbc = read.csv(file.choose(), header=FALSE)

Y = as.matrix(wdbc[,2])
M_index = which(Y[,1] == "M")
Y[M_index,] = 1
Y[-M_index,] = 0
Y = as.matrix(as.numeric(Y))
X = as.matrix(wdbc[,-c(1,2)])
scrub = which(1:ncol(X) %% 3 == 0)
scrub = 11:30
X = X[,-scrub]
X = scale(X) #scale all values in X
glm1 = glm(Y~X, family='binomial')
X = cbind(rep(1),X) #add column of 1 for intercept
M = matrix(1, nrow(X), 1)


### Gradient Calculation ###
Gradient_Cal = function(X, Y, beta, M)
{
  # check dimension
  #if (!(nrow(X) == nrow(Y) == nrow(M)) || !(ncol(X) == nrow(beta))){
  #  "demension mismatch, check data input!"}
  
  W = 1/(1+exp(- (X %*% beta))) #saving W as a vector instead of diag matrix 
  Gradient = t(X) %*% (M*W - Y) #use M*W here to save computation time
  return (Gradient)
}


### Hessian Calculation ###
Hessian_Cal = function(X, beta, M)
{
  W = 1/(1+exp(- (X %*% beta))) #saving W as a vector instead of diag matrix 
  Hession = t(X) %*% diag(as.vector(M*W*(1-W))) %*% X #use M*W*(1-W) here to save computation time
  return (Hession)
}


### Gradient Descent ###
Gradient_Desc = function(X, Y, M, beta_start, step, precision, iter_max)
{
  N = nrow(X)
  P = ncol(X)
  log_ll = c()
  beta = matrix(beta_start, P, 1) #initial value to start
  delta = 1
  iter = 0
  while ( (delta > precision) && (iter < iter_max) )
  { 
    beta_update = beta - step*Gradient_Cal(X, Y, beta, M) #gradient decsent
    delta = sqrt(sum((beta_update - beta)^2))/ sqrt(sum(beta^2)) #calculat absolute error
    X_beta_update = X %*% beta_update #calculate this for likelihood
    log_ll_update = -sum(M*log(1+exp(- X_beta_update))) - sum((M-Y)*X_beta_update) #calculate log likelihood
    log_ll = c(log_ll, log_ll_update) #keep record of log likelihood for convergence
    beta = beta_update #update bets
    iter = iter + 1 #keep track of iteration
  } 
  return (list(iter, beta, log_ll))
}

### Newton's Method ###
Newton_Method = function(X, Y, M, beta_start, step, precision, iter_max)
{
  N = nrow(X)
  P = ncol(X)
  log_ll = c()
  beta = matrix(beta_start, P, 1) #initial value to start
  delta = 1
  iter = 0
  while ( (delta > precision) && (iter < iter_max) )
  { 
    beta_update = beta - solve(Hessian_Cal(X, beta, M)) %*% Gradient_Cal(X, Y, beta, M)
    delta = sqrt(sum((beta_update - beta)^2))/ sqrt(sum(beta^2)) #calculat absolute error
    X_beta_update = X %*% beta_update #calculate this for likelihood
    log_ll_update = -sum(M*log(1+exp(- X_beta_update))) - sum((M-Y)*X_beta_update) #calculate log likelihood
    log_ll = c(log_ll, log_ll_update) #keep record of log likelihood for convergence
    beta = beta_update #update bets
    iter = iter + 1 #keep track of iteration
  }
  return (list(iter, beta, log_ll))
}


#run the algorithm and plot log likelihood to compare
result_gd = Gradient_Desc(X, Y, M, 0, 0.01, 0.000001, 100000)
plot(x = 1:result_gd[[1]], y = result_gd[[3]], type = "l", xlab = "iteration", 
     ylab = "log likelihood", main = "convergence of gradient decsent")

result_nm = Newton_Method(X, Y, M, 0, 0.01, 0.000001, 100000)
plot(x = 1:result_nm[[1]], y = result_nm[[3]], type = "l", xlab = "iteration", 
     ylab = "log likelihood", main = "convergence of Newton's method")



#compare beta with output from glm
Output = matrix(0,12,3)
dimnames(Output) = list(c("num of iter", "intercept", "beta1", "beta2", "beta3", 
                   "beta4", "beta5", "beta6", "beta7", "beta8", "beta9", "beta10"),
                   c("glm", "GD", "NM"))
Output[ ,1] = c(0, as.vector(glm1$coefficients))
Output[ ,2] = c(result_gd[[1]], as.vector(result_gd[[2]]))
Output[ ,3] = c(result_nm[[1]], as.vector(result_nm[[2]]))
Output
