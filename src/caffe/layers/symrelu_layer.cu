#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SymReLUForward(const int n, const Dtype* in, Dtype* out,
    const int channels_, const int height_, const int width_) {
  CUDA_KERNEL_LOOP(index, n) {
    int n = index / (channels_ * height_ * width_);
    out[index + n * channels_ * height_ * width_] = 
      in[index] > 0 ? in[index] : 0;
    out[index + (n + 1) * channels_ * height_ * width_] = 
      in[index] < 0 ? in[index] : 0;
  }
}

template <typename Dtype>
void SymReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // this->Forward_cpu(bottom, top);
  // return;

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SymReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SymReLUBackward(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const int channels_, const int height_, const int width_) {
  CUDA_KERNEL_LOOP(index, n) {
    int n = index / (channels_ * height_ * width_);
    out_diff[index] = 
      in_diff[index + n * channels_ * height_ * width_] * (in_data[index] > 0) + 
      in_diff[index + (n + 1) * channels_ * height_ * width_] * (in_data[index] < 0);
  }
}

template <typename Dtype>
void SymReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // this->Backward_cpu(top, propagate_down, bottom);
  // return;

  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SymReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, channels_, height_, width_);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_CLASS(SymReLULayer);


}  // namespace caffe
