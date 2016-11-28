rm(list=ls())
library(Matrix)


### function to calculate soft threshold ###
soft_threshold = function(y,l)
{
  z = abs(y) - l
  theta = sign(y)*z*(z > 0)
  return (theta)
}


### function to calculate L1_norm ###
L1_norm = function(x)
{
  L1 = sum(abs(x))
  return (L1) 
}


### function to calculate L2_norm ###
L2_norm = function(x)
{
  L2 = sqrt(sum(x^2))
  return (L2) 
}



shrink_function = function(vec, c, tol=1e-10)
{
  #for an input vector vec and a constrain scalar c
  #function returns a shrinked vector with L1 norm equals c
  #by applying soft thresholding to original vector
  
  n = length(vec)
  if (!((1 <= c) && (c <= sqrt(n)))) {
    stop('WARNING: restrict constant is not in correct ranges')
  }
  
  vec_norm = vec / L2_norm(vec)
  if (L1_norm(vec_norm) <= c) {
      return (vec_norm)
   } else {
      delta_max = max(abs(vec))
      delta_min = 0
      delta = 0.5*(delta_max + delta_min)
      temp = soft_threshold(vec, delta)
      vec_delta = temp / L2_norm(temp)
    
      while ( abs(L1_norm(vec_delta) - c) >= tol ){
    
        if(L1_norm(vec_delta) > c){
          delta_min = delta
        }else {
          delta_max = delta
        }
    
        delta = 0.5*(delta_max + delta_min)
        temp = soft_threshold(vec, delta)
    
          #if (sum(temp) == 0) {
          #  vec_delta = temp
          #}else{
            vec_delta = temp / L2_norm(temp)
          #}
      }
      return  (vec_delta) 
    }
}


### PMD(L1, L1) ###
PMD1factor = function(x, c1, c2, tol=1e-10)
{
  n = nrow(x)
  p = ncol(x)
  
  v = rep(1/(sqrt(p)), p)
  u = rep(1/(sqrt(n)), n)
  diff = 1
  
  while ( diff >= tol )
  {
    xv = x %*% v
    temp = xv / L2_norm(xv)
    if ( L1_norm(temp) <= c1 ){
      u_update = temp
    } else {
      u_update = shrink_function(vec=xv, c=c1, tol=tol)
    }
    diff_u = L2_norm(u - u_update)
    u = u_update
    
    xu = crossprod(x, u)
    temp = xu / L2_norm(xu)
    if ( L1_norm(temp) <= c2 ){
      v_update = temp
    } else {
      v_update = shrink_function(vec=xu, c=c2, tol=tol)
    }
    diff_v = L2_norm(v - v_update)
    v = v_update
    
    diff = max(diff_u, diff_v)
  }
    d = t(u) %*% x %*% v
    return (list(u, d, v))
}
  

PMDkfactor = function (mat, k, c1, c2, tol=1e-10)
{
  ##################
  #mat = volcano
  #k = 12
  #c1 = 3.5
  #c2 = 3.5
  #tol = 1e-6
  ##################
  
  if ( (k <= 0) || (round(k) != k)  ){
    stop('WARNING: k needs to be a positive integer')
  }
  
  if ( rankMatrix(mat) < k ){
    stop('WARNING: k can not be greater than the rank of input matrix')
  }
  
  n = nrow(mat)
  p = ncol(mat)
  U = matrix(0, nrow=n, ncol=k)
  D = diag(0, k)
  V = matrix(0, nrow=k, ncol=p)
  X = mat
  
  for (i in 1:k){
    U[,i] = PMD1factor(x=X, c1=c1, c2=c2, tol=tol)[[1]] 
    D[i,i] = PMD1factor(x=X, c1=c1, c2=c2, tol=tol)[[2]]
    V[i, ] = PMD1factor(x=X, c1=c1, c2=c2, tol=tol)[[3]]
    X = X - D[i,i] * U[,i] %*% t(V[i, ])
  }
  mat_hat = U %*% D %*% V
  diff = sum((mat - mat_hat)^2)
  #image(mat_hat)
  
  return (list(U, D, V, mat_hat, diff))
}


### test with simulated data ###
simu_mat = matrix (rnorm(50)+rep(seq(1,5), each=10), nrow=10, ncol=5)

library(readr)
socialdata = read_csv("C:/Users/schen/Dropbox/toChensu/Stats/2016Fall/Big Data/Assignment9/social_marketing.csv")
socialdata = as.matrix(socialdata[,-1])
socialdata = scale(socialdata)

K = 3
c = 0.3
c1 = max(1, c*sqrt(dim(socialdata)[1]))
c2 = max(1, c*sqrt(dim(socialdata)[2]))

result = PMDkfactor(mat=socialdata, k=K, c1=c1, c2=c2, tol=1e-5)
socialdata_U = result[[1]]
socialdata_D = result[[2]]
socialdata_V = result[[3]]
socialdata_hat = result[[4]]

#image(socialdata)
#image(result[[4]])

#for (j in 1:K)
#{
  colnames(socialdata)[socialdata_V[1,] != 0]
  colnames(socialdata)[socialdata_V[2,] != 0]
  colnames(socialdata)[socialdata_V[3,] != 0]
#}

