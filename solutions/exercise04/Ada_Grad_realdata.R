rm(list=ls())
library(Rcpp)
library(RcppEigen)
library(Matrix)
sourceCpp(file="C:/Users/schen/Dropbox/toChensu/Stats/2016Fall/Big Data/Assignment4/Ada_Grad.cpp") 
sourceCpp(file="C:/Users/schen/Dropbox/toChensu/Stats/2016Fall/Big Data/Assignment4/sgdlogit.cpp") 

x=readRDS("C:/Users/schen/Dropbox/toChensu/Stats/2016Fall/Big Data/Assignment4/url_X.rds")
y=readRDS("C:/Users/schen/Dropbox/toChensu/Stats/2016Fall/Big Data/Assignment4/url_y.rds")
M=rep(1, length(y))
x=Matrix(x,sparse=TRUE)
x=t(x)
beta0=rep(0, nrow(x))


time=proc.time()
result_JS = sparsesgd_logit(x,y,M, eta=0.01, npass=1, beta0, 
                            lambda=0.1, discount = 0.01) 
time_JS=proc.time()-time
plot(result_JS$nll_tracker, type="l", log='xy', xlab = "iteration", 
     ylab = "Exp Weighted Avg negative log likelihood", 
     main = "Stochastic Gradient Decsent JS code")


time=proc.time()
result_SC = Ada_Grad_sparse(x,y, beta0, step=0.00000001, npass=1, 
                            alpha=0.01, lambda=0.1) 
time_SC=proc.time()-time
plot(result_SC$nll_ex_avg, type="l", log='xy', xlab = "iteration", 
     ylab = "Exp Weighted Avg negative log likelihood", 
     main = "Stochastic Gradient Decsent SC code")

# compare results of James Scott code with mine code
Output = matrix(0,6,2)
dimnames(Output) = list( c("run time", "intercept", "min beta", 
                           "median beta","mean beta", "max beta"), 
                         c("JS code", "SC code"))
Output[,1] = c(time_JS[3], result_JS$alpha, min(result_JS$beta), 
               median(result_JS$beta), mean(result_JS$beta),  
               max(result_JS$beta))
Output[,2] = c(time_SC[3], result_SC$intercept, min(result_SC$beta), 
               median(result_SC$beta), mean(result_SC$beta),  
               max(result_SC$beta))
Output