#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/reg.hpp"

namespace caffe {

template <typename Dtype>
__global__ void l1_reg_kernel(const int n, const Dtype weight_decay, const Dtype* val, Dtype* diff) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i<n) {
    // version 1
    /*
       if(val[i]-diff[i]>weight_decay)
       diff[i]+=weight_decay;
       else if(val[i]-diff[i]<-weight_decay)
       diff[i]-=weight_decay;
       else
       diff[i] = val[i];
     */

    // version 2
    if(val[i] - diff[i] > 1e-4) {
      diff[i] += weight_decay;
     } else if(val[i] - diff[i] < -1e-4) {
      diff[i] -= weight_decay;
     } else {
      diff[i] = val[i];
     }
  }
}

template <>
void caffe_l1_reg<float>(const int n,const float weight_decay, const float* val_gpu, float* diff_gpu) {
  l1_reg_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,weight_decay,val_gpu,diff_gpu);
}

template <>
void caffe_l1_reg<double>(const int n,const double weight_decay, const double* val_gpu, double* diff_gpu) {
  l1_reg_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,weight_decay,val_gpu,diff_gpu);
}

}  // namespace caffe
