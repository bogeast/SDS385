//
//  Ada_Grad.cpp
//  
//
//  Created by Su Chen on 9/26/16.
//
//

#include <iostream>
#include <RcppEigen.h>
#include <Rcpp.h>
#include <cmath>
#include <algorithm>    // std::max


using namespace Rcpp;
using namespace std;
using namespace Eigen;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::MatrixXi;
using Eigen::Upper;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::SparseVector;
typedef Eigen::MappedSparseMatrix<double>  MapMatd;
typedef Map<MatrixXi>  MapMati;
typedef Map<VectorXd>  MapVecd;
typedef Map<VectorXi>  MapVeci;


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
SEXP Ada_Grad_sparse(MapMatd x, VectorXd y, VectorXd beta0, double step, int npass,  double alpha=0.01,  double lambda=0.1) {
    int P = x.rows(); // number of features
    int N = x.cols(); // number of observations
    
    // define and initialize all the variables
    double w_hat = (y.sum() + 1.0) / (N + 2.0); // add little smoothing to avoid log0
    double intercept = log( w_hat / (1.0 - w_hat) ); 
      
    double xb, exb, y_hat, grad0, grad, skip;
    double nll = 0.0;
    int j = 0;
    int k = 0;

    VectorXd beta(P);
    VectorXd G(P);
    for(j=0; j<P; j++) {
      beta(j) = beta0(j); // initial beta value
      G(j) = 1e-3; // initial G includes a fudge factor
    } 

    double G0 = 1e-3; // initial G0 for intercept

    NumericVector nll_ex_avg(npass*N, 0.0);
    //nll_ex_avg[0]=log(2); // initial nll for beta = 0

    SparseVector<double> xi(P); // ith observation vector
    NumericVector last_update(P, 0.0); // keep track of which feature being updated

    for (int pass=0; pass<npass; pass++){

      for (int i=0; i<N; i++) {

          xi = x.innerVector(i);
          xb = xi.dot(beta) + intercept;
          exb = exp(xb);
          y_hat = exb / (1.0 + exb);

          // update negative log likelihood
          nll = alpha*( log(1.0+exb) - y(i)*xb ) + (1.0-alpha)*nll;
          nll_ex_avg(k) = nll;
          
          // update intercept
          grad0 = y_hat - y[i]; // gradient=(y_hat-y[i])*xi, xi=1 for intercept
          G0 = G0 + grad0*grad0;
          intercept = intercept - step/sqrt(G0)*grad0;
          
          
          for (SparseVector<double>::InnerIterator it(xi); it; ++it) {
            //Rcout<<"it="<<it<<std::endl;
            
              j = it.index();
              skip = k - last_update(j);
              // update all the L2 penalty terms since last update
              beta(j) = beta(j) - step*skip*2.0*lambda*beta(j)/sqrt(G(j));
              // calculate gradient for feature j
              grad = grad0*it.value();
              // update jth element of G
              G(j) = G(j) + grad*grad;
              // update beta(j) 
              //beta(j) = beta(j) - step*(grad+2.0*lambda*beta(j))/ sqrt( G(j) ); 
              beta(j) = beta(j) - step*grad / sqrt( G(j) );
              // update penalty for this iteration
              beta(j) = beta(j) - step*2.0*lambda*beta(j) / sqrt( G(j) ); 
              
              // update last_update vector
              last_update(j) = k;
          }
          // total iteration tracker
          k = k + 1;
         
      }
    }
    
    // in the end need to apply the accumulated penalty
    for(int j=0; j< P; j++) {
      // but only for those features that didn't get updated in the last iteration
      if ( k > last_update(j) ) {
          skip = k - last_update(j);
          beta(j) = beta(j) - step*skip*2.0*lambda*beta(j)/sqrt(G(j));
        }
    }
    
    return Rcpp::List::create(Rcpp::Named("intercept") = intercept,
                             Rcpp::Named("beta") = beta,
                             Rcpp::Named("nll_ex_avg") = nll_ex_avg);
}




