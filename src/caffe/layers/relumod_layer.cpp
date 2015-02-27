#include <algorithm>
#include <vector>
#include <stdio.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReLUModLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_sample_ = 0;
  num_pos_.Reshape(1, bottom[0]->height(), bottom[0]->width(), bottom[0]->channels());
  caffe_set(num_pos_.count(), (unsigned)0, num_pos_.mutable_cpu_data());  
}

template <typename Dtype>
void ReLUModLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLUModLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}

template <typename Dtype>
void ReLUModLayer<Dtype>::Analysis(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  unsigned* num_pos_data = num_pos_.mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    int offset = n * num_pos_.count();
    for (int i = 0; i < num_pos_.count(); ++i) {
      if (bottom_data[offset + i] > 0) {
        num_pos_data[i] ++;
      }
    }
    num_sample_ ++;
  }
}

template <typename Dtype>
void ReLUModLayer<Dtype>::PrintAnalysis() {
  FILE* pfile;
  string filename = this->layer_param_.name() + "-analysis";
  pfile = fopen(filename.c_str(), "a+");
  const unsigned* num_pos_data = num_pos_.cpu_data();
  for (int i = 0; i < num_pos_.count(); ++i) {
    fprintf(pfile, "%f\n", double(num_pos_data[i]) / double(num_sample_));
  }
  fclose(pfile);
}

template <typename Dtype>
void ReLUModLayer<Dtype>::Print(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>* top) {
  FILE* pfile;
  pfile = fopen(this->layer_param_.name().c_str(), "a+");
  const Dtype* top_data = (*top)[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (bottom[0]->height() != 13) {
    return;
  }
  const int count = bottom[0]->count();
  // LOG(INFO) << "ReLUModLayer::Print()";
  for (int i = 0; i < count; ++i) {
    /*
    if (top_data[i] == 0) {
      fprintf(pfile, "0 ");
    } else {
      fprintf(pfile, "1 ");
    }
    */
    // fprintf(pfile, "%e ", bottom_data[i]);
  }
  fprintf(pfile, "\n");
  fclose(pfile);
}

#ifdef CPU_ONLY
STUB_GPU(ReLUModLayer);
#endif

INSTANTIATE_CLASS(ReLUModLayer);

}  // namespace caffe
