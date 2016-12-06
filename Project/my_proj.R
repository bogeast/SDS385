rm(list=ls())
library(pROC)

source("C:/Users/schen/Dropbox/toChensu/Stats/2016Fall/Big Data/Project/my_sgd.R")
source("C:/Users/schen/Dropbox/toChensu/Stats/2016Fall/Big Data/Project/my_cmc.R")


defaultdata = read.csv("C:/Users/schen/Dropbox/toChensu/Stats/2016Fall/Big Data/Project/default.csv",header=TRUE)
N = dim.data.frame(defaultdata)[1]
P = dim.data.frame(defaultdata)[2]
defaultX = as.matrix(cbind(rep(1,N), defaultdata[,2:6], scale(defaultdata[,7:(P-1)])))
defaultY = as.matrix(defaultdata[,P])
test_index = sample(N, round(0.05*N,0))
testX = defaultX[test_index,]
testY = defaultY[test_index]
trainX = defaultX[-test_index,]
trainY = defaultY[-test_index]
M = matrix(1, length(trainY), 1)


### run glm and save output for comparision ###
glmfit = glm(trainY~trainX[,-1], family='binomial')
beta_glm = as.vector(glmfit$coefficients)
nll_glm = Neg_ll(beta_glm, trainX, trainY,  M)


### run sgd and cmc ###
Iter_Max = 100000
ptm = proc.time()
result_sgd = Sgd_Ada_Grad(x=trainX, y=trainY, m=M, step=1, lambda = 0.01, iter_max=Iter_Max)
proc.time() - ptm
beta_sgd = result_sgd[[2]][,Iter_Max]

ptm = proc.time()
result_cmc = Cmc_Probit(x=trainX, y=trainY, s=100, burnin=5000, mcmc=10000, mu0=0, tau0=0.01)
proc.time() - ptm
beta_cmc = rowMeans(result_cmc) 
beta_cmc_sd = apply(result_cmc, 1, sd)
beta_cmc_L = beta_cmc - 2*beta_cmc_sd
beta_cmc_U = beta_cmc + 2*beta_cmc_sd


### plot negative log likelihood for sgd ###
plot_x = 1:result_sgd[[1]]
plot(x=plot_x, y=result_sgd[[3]], col="red",
     type="l", xlab="iteration", ylab="negative log likelihood", 
     main="Convergence of Stochastic Gradient Descent")
lines(x=plot_x, y=result_sgd[[4]], col="yellow")
lines(x=plot_x, y=result_sgd[[5]], col="green")
lines(x=c(1,Iter_Max), y=rep(nll_glm, 2), col="black")
legend("topright", cex=.5, c("full neg llog","running avg", "exp weighted avg","glm output"), 
       col=c("red","yellow","green","black"), lty=c(1,1,1,1))


### prediction ###
Ntest = length(testY)

logit_glm = 1/(1+exp(- (testX %*% beta_glm)))
predict_glm = round(logit_glm, 0)
prederr_glm = sum((predict_glm - testY)^2)/Ntest
roc_glm = roc(testY, as.vector(logit_glm))

logit_sgd = 1/(1+exp(- (testX %*% beta_sgd)))
predict_sgd = round(logit_sgd, 0)
prederr_sgd = sum((predict_sgd - testY)^2)/Ntest
roc_sgd = roc(testY, as.vector(logit_sgd))

probit_cmc = dnorm(testX %*% beta_cmc)
predict_cmc = round(probit_cmc, 0)
prederr_cmc = sum((predict_cmc - testY)^2)/Ntest
roc_cmc = roc(testY, as.vector(probit_cmc))


### print results and plot roc###
Output = matrix(0, (ncol(trainX)+2), 5)
dimnames(Output) = list(c("intercept", colnames(trainX[,-1]), "predict error rate", "AUC"),
                        c("glm", "sgd", "cmc", "cmc 95% CIL", "cmc 95% CIU"))
Output[,1] = c(beta_glm, prederr_glm, auc(roc_glm))
Output[,2] = c(beta_sgd, prederr_sgd, auc(roc_sgd))
Output[,3] = c(beta_cmc, prederr_cmc, auc(roc_cmc))
Output[,4] = c(beta_cmc_L, NA, NA)
Output[,5] = c(beta_cmc_U, NA, NA)
Output


### plot beta estimates from glm, sgd and cmc ###
beta_length = length(beta_glm)
Beta_Output = Output[1:beta_length, ]
plot(x=rep(1:beta_length,2), y=c(Beta_Output[,4], Beta_Output[,5]), 
     xlab="Betas", ylab="Beta estimates", xlim=c(1,beta_length), ylim=c(min(Beta_Output),max(Beta_Output)), 
     main="95% Confidence Interval of Beta estimates",
     type="p", pch=4, col="green")
points(x=1:beta_length, y=Beta_Output[,1], pch=19, col="black")
points(x=1:beta_length, y=Beta_Output[,2], pch=19, col="red")
points(x=1:beta_length, y=Beta_Output[,3], pch=19, col="green")
segments(x0=1:beta_length, x1=1:beta_length, y0=Beta_Output[,4], Beta_Output[,5], col="green")
legend("bottomright", c("glm","sgd", "cmc", "cmc 95% CI"), col=c("black","red","green", "green"), pch=c(19,19,19,4))


### plot ROC ###
plot(roc_glm, main="ROC plot")
lines(x=roc_sgd$specificities, y=roc_sgd$sensitivities, col="red", lwd=2)
lines(x=roc_cmc$specificities, y=roc_cmc$sensitivities, col="green", lwd=2)
legend("topleft", c("glm: AUC=0.721","sgd: AUC=0.714", "cmc: AUC=0.707"), col=c("black","red","green"), lty=c(1,1,1))


### plot trace plot for cmc ###
par(mfrow=c(3,3))
plot(result_cmc[1, ], type="l", ylab="intercept")
plot(result_cmc[2, ], type="l", ylab=colnames(trainX)[2])
plot(result_cmc[3, ], type="l", ylab=colnames(trainX)[3])
plot(result_cmc[4, ], type="l", ylab=colnames(trainX)[4])
plot(result_cmc[6, ], type="l", ylab=colnames(trainX)[6])
plot(result_cmc[7, ], type="l", ylab=colnames(trainX)[7])
plot(result_cmc[14, ], type="l", ylab=colnames(trainX)[14])
plot(result_cmc[15, ], type="l", ylab=colnames(trainX)[15])
plot(result_cmc[21, ], type="l", ylab=colnames(trainX)[21])

