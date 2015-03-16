#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReCUForward(const int n, const Dtype* bottom, Dtype* top,
    Dtype negative_slope, Dtype xc, Dtype yc, Dtype s, Dtype ys, Dtype beta);

template <>
__global__ void ReCUForward<float>(const int n, const float* bottom, float* top,
    float negative_slope, float xc, float yc, float s, float ys, float beta) {
  CUDA_KERNEL_LOOP(index, n) {
    if (bottom[index] <= 0) {
      top[index] = negative_slope * bottom[index];
    }
    else if (bottom[index] <= s) {
      top[index] = sqrtf(xc * xc + yc * yc - (bottom[index] - xc) * (bottom[index] - xc)) + yc;
    }
    else {
      top[index] = ys + (bottom[index] - s) * beta;
    }   
  }
}
template <>
__global__ void ReCUForward<double>(const int n, const double* bottom, double* top,
    double negative_slope, double xc, double yc, double s, double ys, double beta) {
  CUDA_KERNEL_LOOP(index, n) {
    if (bottom[index] <= 0) {
      top[index] = negative_slope * bottom[index];
    }
    else if (bottom[index] <= s) {
      top[index] = sqrt(xc * xc + yc * yc - (bottom[index] - xc) * (bottom[index] - xc)) + yc;
    }
    else {
      top[index] = ys + (bottom[index] - s) * beta;
    }   
  }
}
template <typename Dtype>
void ReCULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  if (Caffe::phase() == Caffe::TEST) {
    // this->Analysis(bottom, top);
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.recu_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReCUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope, xc, yc, s, ys, beta);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReCUBackward(const int n, const Dtype* top_diff,
    const Dtype* bottom_data, Dtype* bottom_diff, Dtype negative_slope, Dtype xc, Dtype yc, Dtype s, Dtype ys, Dtype beta);

template <>
__global__ void ReCUBackward<float>(const int n, const float* top_diff,
    const float* bottom_data, float* bottom_diff, float negative_slope, float xc, float yc, float s, float ys, float beta) {
  CUDA_KERNEL_LOOP(index, n) {
    if (bottom_data[index] <= 0) {
      bottom_diff[index] = negative_slope * top_diff[index];
    }
    else if (bottom_data[index] <= s) {
      bottom_diff[index] = top_diff[index] * (xc - bottom_diff[index]) / sqrtf(-bottom_diff[index] * bottom_diff[index] + 2 * xc * bottom_diff[index] + yc * yc);
    }
    else {
      bottom_diff[index] = top_diff[index] * beta;
    }   
  }
}

template <>
__global__ void ReCUBackward<double>(const int n, const double* top_diff,
    const double* bottom_data, double* bottom_diff, double negative_slope, double xc, double yc, double s, double ys, double beta) {
  CUDA_KERNEL_LOOP(index, n) {
    if (bottom_data[index] <= 0) {
      bottom_diff[index] = negative_slope * top_diff[index];
    }
    else if (bottom_data[index] <= s) {
      bottom_diff[index] = top_diff[index] * (xc - bottom_diff[index]) / sqrt(-bottom_diff[index] * bottom_diff[index] + 2 * xc * bottom_diff[index] + yc * yc);
    }
    else {
      bottom_diff[index] = top_diff[index] * beta;
    }   
  }
}

template <typename Dtype>
void ReCULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    Dtype negative_slope = this->layer_param_.recu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReCUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope, xc, yc, s, ys, beta);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_CLASS(ReCULayer);


}  // namespace caffe
