#include "LDshrinkCUDA/calc_cov.h"
//#include "Utilities.cuh"
//#include "TimingGPU.cuh"
#include <RcppEigen.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>


typedef thrust::host_vector<double, thrust::cuda::experimental::pinned_allocator<double> > pinnedVector;




/*************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX */
/*************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
  
T Ncols; // --- Number of columns
  
__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}
  
__host__ __device__ T operator()(T i) { return i / Ncols; }
};

template <typename T>
struct linear_index_to_col_index : public thrust::unary_function<T,T> {
  
T Ncols; // --- Number of columns
  
__host__ __device__ linear_index_to_col_index(T Ncols) : Ncols(Ncols) {}
  
__host__ __device__ T operator()(T i) { return i % Ncols; }
};




template <typename T>
struct k_to_row : public thrust::unary_function<T,T> {
  
T NX; // --- Number of columns
  
__host__ __device__ k_to_row(T NX) : NX(NX) {}
  
__host__ __device__ T operator()(T i) { return NX-2-floor(sqrt((double)-8*i+4*NX*(NX-1)-7)/2.0 -0.5); }
};

template <typename T>
struct k_to_col : public thrust::unary_function<T,T> {
  
T NX; // --- Number of columns
  
__host__ __device__ k_to_col(T NX) : NX(NX) {}
  
__host__ __device__ T operator()(T k) {
int i= NX-2-floor(sqrt((double)-8*k+4*NX*(NX-1)-7)/2.0 -0.5);
return k + i + 1 - NX*(NX-1)/2.0 + (NX-i)*((NX-i)-1)/2.0;
}
};


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

        
  

__global__ void ldshrink(float* S,
			 float* map,
			 float* diag,
			 int NX,
			 float m,
			 float Ne,
			 float cutoff,
			 float theta){
  



  unsigned int i= blockIdx.x*blockDim.x+threadIdx.x;
  
  unsigned int j= blockIdx.y*blockDim.y+threadIdx.y;

  unsigned int k=j*NX+i;
  
  if((i<NX)&&(j<NX)){
      
    // printf("threadIdx.x: %i blockIdx.x: %i i:%i\n",threadIdx.x,blockIdx.x,i);
    // printf("threadIdx.y: %i blockIdx.y: %i j:%i\n",threadIdx.y,blockIdx.y,i);
      
      
    float tshrinkage=0;
    if(i!=j){
      float rho=-((4*Ne*(abs(map[j]-map[i])))/100)/(2*m);
      tshrinkage=exp(rho);
      if(tshrinkage<cutoff){
	tshrinkage=0;
      }
    }else{
      tshrinkage=1;
    }
    S[k]=tshrinkage*S[k];
    S[k]=((1-theta)*(1-theta))*S[k];
    if(i==j){
      S[k]+=(0.5*theta)*(1-0.5*theta);
      diag[i]=sqrt(S[k]);
    }
  }
}
  
struct CastToFloat
{
  float operator()(double value) const { return static_cast<float>(value);}
};

struct CastToDouble
{
  double operator()(float value) const { return static_cast<double>(value);}
};



void gen_csr(thrust::device_vector<float> &d_mat,
		  const size_t r,
		  const size_t c,
	     int &nnzm,
	     thrust::device_vector<int> &nnzr,
	     thrust::device_vector<float> &csrval,
	     thrust::device_vector<int> &csrrowptr,
	     thrust::device_vector<int> &csrcolind){





  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseStatus_t status;
  
  cusparseMatDescr_t descr=0;
  
  status= cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO); 
    // cusparseSnnz(cusparseHandle_t handle,
    // 		 cusparseDirection_t dirA,
    // 		 int m,
    // 		 int n,
    // 		 const cusparseMatDescr_t descrA,
    // 		 const float *A,
    // 		 int lda,
    // 		 int *nnzPerRowColumn,
    // 		 int *nnzTotalDevHostPtr)
  cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;
  nnzr.resize(r);
  if(nnzm<0){
    cusparseSnnz(handle,dirA,r,c,descr,
		 thrust::raw_pointer_cast(d_mat.data()),
		 r,thrust::raw_pointer_cast(nnzr.data()),&nnzm);
    csrval.resize(nnzm);
    csrrowptr.resize(r+1);
    csrcolind.resize(nnzm);
  }
  
  //  cusparseSdense2csc(cusparseHandle_t handle,
  // int m, rownum
  // int n, colnum 
  // const cusparseMatDescr_t descrA, CUSPARSE_MATRIX_TYPE_GENERAL 
  // const float *A, array of dimension (lda,n)
  // int lda, leading dimension
  // const int *nnzPerCol,
  // float *csrValA,
  // int *csrRowPtrA,
  // int *csrColIndA)
  cusparseSdense2csr(handle,
		     r,c,descr,
		     thrust::raw_pointer_cast(d_mat.data()),
		     r,thrust::raw_pointer_cast(nnzr.data()),
		     thrust::raw_pointer_cast(csrval.data()),
		     thrust::raw_pointer_cast(csrrowptr.data()),
		     thrust::raw_pointer_cast(csrcolind.data()));

}



void gen_csc(thrust::device_vector<float> &d_mat,
	     const size_t r,
	     const size_t c,
	     int &nnzm,
	     thrust::device_vector<int> &nnzc,
	     thrust::device_vector<float> &cscval,
	     thrust::device_vector<int> &cscrowind,
	     thrust::device_vector<int> &csccolptr){



  cusparseStatus_t status;
  
  cusparseMatDescr_t descr=0;
  
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;
  // cusparseSnnz(cusparseHandle_t handle,
  // 		 cusparseDirection_t dirA,
  // 		 int m,
  // 		 int n,
  // 		 const cusparseMatDescr_t descrA,
  // 		 const float *A,
  // 		 int lda,
  // 		 int *nnzPerRowColumn,
  // 		 int *nnzTotalDevHostPtr)
  // cusparseStatus_t status;
  
  // cusparseMatDescr_t descr=0;
  
  status= cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO); 


  nnzc.resize(c);
  if(nnzm<0){
    cusparseSnnz(handle,
		 dirA,r,c,descr,
		 thrust::raw_pointer_cast(d_mat.data()),
		 r,thrust::raw_pointer_cast(nnzc.data()),&nnzm);
    cscval.resize(nnzm);
    cscrowind.resize(nnzm);
    csccolptr.resize(c+1);
  }
  
  //  cusparseSdense2csc(cusparseHandle_t handle,
  // int m, rownum
  // int n, colnum 
  // const cusparseMatDescr_t descrA, CUSPARSE_MATRIX_TYPE_GENERAL 
  // const float *A, array of dimension (lda,n)
  // int lda, leading dimension
  // const int *nnzPerCol,
  // float *csrValA,
  // int *csrRowPtrA,
  // int *csrColIndA)
  cusparseSdense2csc(handle,
		     r,c,descr,
		     thrust::raw_pointer_cast(d_mat.data()),
		     r,thrust::raw_pointer_cast(nnzc.data()),
		     thrust::raw_pointer_cast(cscval.data()),
		     thrust::raw_pointer_cast(cscrowind.data()),
		     thrust::raw_pointer_cast(csccolptr.data()));
}
  



void cuda_cov(thrust::device_vector<float> &d_X,thrust::device_vector<float> &d_cov,const size_t N, const size_t p){

  
  cublasHandle_t handle;
  cublasCreate(&handle);

  /*************************************************/
  /* CALCULATING THE MEANS OF THE RANDOM VARIABLES */
  /*************************************************/
  // --- Array containing the means multiplied by Nsamples
  thrust::device_vector<float> d_means(p);

  thrust::device_vector<float> d_ones(N, 1.f);

  float alpha = 1.f / (float)N;
  float beta  = 0.f;
  cublasSgemv(handle, CUBLAS_OP_T, N, p,
	      &alpha, thrust::raw_pointer_cast(d_X.data()), N, 
	      thrust::raw_pointer_cast(d_ones.data()), 1, &beta,
	      thrust::raw_pointer_cast(d_means.data()), 1);

  /**********************************************/
  /* SUBTRACTING THE MEANS FROM THE MATRIX ROWS */
  /**********************************************/
  thrust::transform(
		    d_X.begin(), d_X.end(),
		    thrust::make_permutation_iterator(d_means.begin(),
						      thrust::make_transform_iterator(thrust::make_counting_iterator(0),
										      linear_index_to_row_index<int>(N))),
		    d_X.begin(),
		    thrust::minus<float>());    
  
  /*************************************/
  /* CALCULATING THE COVARIANCE MATRIX */
  /*************************************/
  

    

  
  
  alpha = 1.f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, N, &alpha,
	      thrust::raw_pointer_cast(d_X.data()), N, thrust::raw_pointer_cast(d_X.data()), N, &beta,
	      thrust::raw_pointer_cast(d_cov.data()), p);
  
  // --- Final normalization by N - 1

  thrust::transform(
		    d_cov.begin(), d_cov.end(),
		    thrust::make_constant_iterator((float)(N-1)),
		    d_cov.begin(),
		    thrust::divides<float>());

    
}

void cuda_cov2ld(thrust::device_vector<float> &d_cov,thrust::device_vector<float> &d_map,thrust::device_vector<float> &d_diag,const size_t p,const ldp ldparams){
  
  float m,ne,cutoff,theta;
  std::tie(m,ne,cutoff,theta) = ldparams;

    dim3 threadsPerBlock(16,16);
  dim3 numBlocks(ceil((float)p/(float)16.0)+1,ceil((float)p/(float)16)+1);

  int blocksx=ceil(p/threadsPerBlock.x);
  int blocksy=ceil(p/threadsPerBlock.y);

      
  ldshrink<<<numBlocks,threadsPerBlock>>>(thrust::raw_pointer_cast(d_cov.data()),
					  thrust::raw_pointer_cast(d_map.data()),
					  thrust::raw_pointer_cast(d_diag.data()),
					  p,
					  m,
					  ne,
					  cutoff,
					  theta);
  cudaDeviceSynchronize();

  thrust::transform(
		    d_cov.begin(), d_cov.end(),
		    thrust::make_permutation_iterator(d_diag.begin(),
						      thrust::make_transform_iterator(thrust::make_counting_iterator(0),
										      linear_index_to_row_index<int>(p))),
		    d_cov.begin(),
		    thrust::divides<float>());

  thrust::transform(d_cov.begin(), d_cov.end(),
		    thrust::make_permutation_iterator(d_diag.begin(),
						      thrust::make_transform_iterator(thrust::make_counting_iterator(0),
										      linear_index_to_col_index<int>(p))),
		    d_cov.begin(),
		    thrust::divides<float>());
    
    

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaDeviceSynchronize();
}


void cuda_calcld(const float* X, const float* map,float* ret_cov,const size_t p,const size_t N,const ldp ldparams){
 
  thrust::device_vector<float> d_X(N*p);
  thrust::device_vector<float> d_map(p);
  
  thrust::device_vector<float> d_cov(p * p);
  thrust::device_vector<float> d_diag(p);
  thrust::copy(&X[0],&X[0]+N*p,d_X.begin());
  thrust::copy(&map[0],&map[p],d_map.begin());
  //  cudaMemcpy(thrust::raw_pointer_cast(d_X.data()),X,p*p*sizeof(float),cudaMemcpyHostToDevice);
  //  cudaMemcpy(thrust::raw_pointer_cast(d_map.data()),map,p*sizeof(float),cudaMemcpyHostToDevice);
  
  // --- cuBLAS handle creation


  cuda_cov(d_X,d_cov,N,p);


  cuda_cov2ld(d_cov,d_map,d_diag,p,ldparams);
    

  cudaMemcpy(ret_cov,thrust::raw_pointer_cast(d_cov.data()),p*p*sizeof(float),cudaMemcpyDeviceToHost);    
 

}

void cuda_calcld_sp(const float* X, const float* map,std::vector<int> &outerIndexPtr,
		    std::vector<int> &innerIndices, std::vector<float> &values,const size_t p,const size_t N, int &nnz,const ldp ldparams){

  // int outerIndexPtr[cols+1];
  // int innerIndices[nnz];
  // double values[nnz];
  // Map<SparseMatrix<double> > sm1(rows,cols,nnz,outerIndexPtr, // read-write
  //                                innerIndices,values);

  thrust::device_vector<float> d_X(N*p);
  thrust::device_vector<float> d_map(p);
  
  thrust::device_vector<float> d_cov(p * p);
  thrust::device_vector<float> d_diag(p);
  thrust::copy(&X[0],&X[0]+N*p,d_X.begin());
  thrust::copy(&map[0],&map[p],d_map.begin());
  //  cudaMemcpy(thrust::raw_pointer_cast(d_X.data()),X,p*p*sizeof(float),cudaMemcpyHostToDevice);
  //  cudaMemcpy(thrust::raw_pointer_cast(d_map.data()),map,p*sizeof(float),cudaMemcpyHostToDevice);
  
  // --- cuBLAS handle creation


  cuda_cov(d_X,d_cov,N,p);


  cuda_cov2ld(d_cov,d_map,d_diag,p,ldparams);
  nnz=-1;
  thrust::device_vector<int> nnzr(0);
  thrust::device_vector<float> cscval(0);
  thrust::device_vector<int> cscrowind(0);
  thrust::device_vector<int> csccolptr(0);
  gen_csc(d_cov,p,p,nnz,nnzr,cscval,cscrowind,csccolptr);
  outerIndexPtr.resize(cscrowind.size());
  innerIndices.resize(csccolptr.size());
  values.resize(cscval.size());

  
  
  

  cudaMemcpy(outerIndexPtr.data(),thrust::raw_pointer_cast(cscrowind.data()),cscrowind.size()*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(innerIndices.data(),thrust::raw_pointer_cast(csccolptr.data()),csccolptr.size()*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(values.data(),thrust::raw_pointer_cast(cscval.data()),cscval.size()*sizeof(float),cudaMemcpyDeviceToHost);


}
