#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SymReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  for (int top_id = 0; top_id < top->size(); ++top_id) {
    (*top)[top_id]->Reshape(num_, channels_ * 2, height_, width_);
  }
}
 
template <typename Dtype>
void SymReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          top_data[w + width_ * (h + height_ * (c + 2 * channels_ * n))] = 
            std::max(bottom_data[w + width_ * (h + height_ * (c + channels_ * n))], Dtype(0));
          top_data[w + width_ * (h + height_ * (c + channels_ + 2 * channels_ * n))] = 
            std::min(bottom_data[w + width_ * (h + height_ * (c + channels_ * n))], Dtype(0));
        }
      }
    }
  }
}

template <typename Dtype>
void SymReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            bottom_diff[w + width_ * (h + height_ * (c + channels_ * n))] = 
              (bottom_data[w + width_ * (h + height_ * (c + channels_ * n))] > 0) * 
              top_diff[w + width_ * (h + height_ * (c + 2 * channels_ * n))] + 
              (bottom_data[w + width_ * (h + height_ * (c + channels_ * n))] < 0) * 
              top_diff[w + width_ * (h + height_ * (c + channels_ + 2 * channels_ * n))];
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SymReLULayer);
#endif

INSTANTIATE_CLASS(SymReLULayer);

}  // namespace caffe
