#ifndef CALC_COV_H
#define CALC_COV_H
#include <tuple>
#include <vector>

enum LDind{
  M,
  NE,
  CUTOFF,
  THETA
};


typedef std::tuple<float,float ,float,float> ldp;


void cuda_calcld(const float* X, const float* map,float* ret_cov,const size_t p,const size_t N,const ldp ldparams);

void cuda_calcld_sp(const float* X, const float* map,std::vector<int> &outerIndexPtr,
		    std::vector<int> &innerIndices, std::vector<float> &values,const size_t p,const size_t N, int &nnz,const ldp ldparams);


#endif
