#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/reg.hpp"

namespace caffe {

template <typename Dtype>
void caffe_l1_reg_group(const Dtype lambda, const int GROUP_, Blob<Dtype>& val_blob, Blob<Dtype>& diff_blob) {
  Dtype* val = val_blob.mutable_cpu_data();
  Dtype* diff = diff_blob.mutable_cpu_data();
  int N_ = val_blob.channels() * val_blob.height() * val_blob.width();
  int NUM_OUTPUT_ = val_blob.num();
  for (int g = 0; g < GROUP_; ++g) {
    for (int c = 0; c < N_; ++c) {
      Dtype norm = 0;
      Dtype value;
      for (int y = 0; y < NUM_OUTPUT_ / GROUP_; ++y) {
        value = val[c + N_ * (y + g * NUM_OUTPUT_ / GROUP_)];
        norm += value * value;
      }
      norm = sqrt(norm);
      if (norm > 1e-4) {
        norm = lambda / norm;
        for (int y = 0; y < NUM_OUTPUT_ / GROUP_; ++y) {
          diff[c + N_ * (y + g * NUM_OUTPUT_ / GROUP_)] += val[c + N_ * (y + g * NUM_OUTPUT_ / GROUP_)] * norm;
        }
      }
    }
  }
  val = val_blob.mutable_gpu_data();
  diff = diff_blob.mutable_gpu_data();
}

template
void caffe_l1_reg_group<float>(float lambda, const int GROUP_, Blob<float>& val_blob, Blob<float>& diff_blob);
template
void caffe_l1_reg_group<double>(double lambda, const int GROUP_, Blob<double>& val_blob, Blob<double>& diff_blob);

}  // namespace caffe
