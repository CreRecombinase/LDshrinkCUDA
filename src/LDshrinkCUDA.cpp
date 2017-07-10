// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "LDshrinkCUDA.h"
#include <tuple>



double calc_nmsum(const double m){
  int msize=(2*(int)m-1);
  Eigen::ArrayXd tx(msize);
  tx.setLinSpaced(msize,1,(int)(2*m-1));
  return  (1/tx).sum();
}

//[[Rcpp::export]]
double calc_theta(const double m){
  double nmsum=calc_nmsum(m);
  return((1/nmsum)/(2*m+1/nmsum));
}







//[[Rcpp::export]]
Eigen::MatrixXd calcLD_CUDA(const Eigen::MatrixXd &hmata,const Eigen::ArrayXd &map,const double m=85, const double Ne=11490.672741, const double cutoff=1e-3){


  size_t n=hmata.rows();

  Eigen::MatrixXf hmataf = hmata.cast<float>();
  Eigen::MatrixXf hmapaf = map.cast<float>();

  size_t p=map.size();

  
  
  double theta=calc_theta(m);
  Eigen::MatrixXf ret_cov(p,p);
  ldp ldparams = std::make_tuple(m,Ne,cutoff,theta);
  cuda_calcld(hmataf.data(),hmapaf.data(),ret_cov.data(),p,n,ldparams);
  return(ret_cov.cast<double>());
}


//[[Rcpp::export]]
Rcpp::List calcLD_CUDA_sp(const Eigen::MatrixXd &hmata,const Eigen::ArrayXd &map,const double m=85, const double Ne=11490.672741, const double cutoff=1e-3){

  using namespace Rcpp;
  size_t n=hmata.rows();

  Eigen::MatrixXf hmataf = hmata.cast<float>();
  Eigen::MatrixXf hmapaf = map.cast<float>();

  size_t p=map.size();

  
  
  double theta=calc_theta(m);
  Eigen::MatrixXf ret_cov(p,p);
  ldp ldparams = std::make_tuple(m,Ne,cutoff,theta);
  std::vector<int> outerIndexPtr(0);
  std::vector<int> innerIndices(0);
  std::vector<float> values(0);
  int nnz;
  
  cuda_calcld_sp(hmataf.data(),hmapaf.data(),outerIndexPtr,innerIndices,values,p,n,nnz,ldparams);
  return(Rcpp::List::create(_["outerIndex"] = outerIndexPtr,
			    _["innerIndices"] = innerIndices,
			    _["values"] = values,
			    _["nnz"] = nnz));
}
  


  






