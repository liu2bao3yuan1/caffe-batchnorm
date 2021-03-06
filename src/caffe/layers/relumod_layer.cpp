#include <algorithm>
#include <vector>
#include <stdio.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReLUModLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  hist_res = 256;
  num_sample_ = 0;
  num_pos_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  sum_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  sum_sq_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  hist_.Reshape(1, 1, hist_res * 2 + 1, bottom[0]->channels());
  sum_prod_.Reshape(1, 1, bottom[0]->channels(), bottom[0]->channels());

  caffe_set(sum_.count(), (Dtype)0, sum_.mutable_cpu_data());  
  caffe_set(sum_sq_.count(), (Dtype)0, sum_sq_.mutable_cpu_data());  
  caffe_set(num_pos_.count(), (unsigned)0, num_pos_.mutable_cpu_data());  
  caffe_set(hist_.count(), (unsigned)0, hist_.mutable_cpu_data());  
  caffe_set(sum_prod_.count(), (Dtype)0, sum_prod_.mutable_cpu_data());  

  string filename = this->layer_param_.name() + "-analysis";
  string cmd = "rm " + filename;
  system(cmd.c_str());
  LOG(INFO) << "ReLUMod: LayerSetUp";
}

template <typename Dtype>
void ReLUModLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
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
  if (sum_.height() == 1 && sum_.width() == 1) { // only for conv layers
    return;
  }
  int height_ = bottom[0]->height(); 
  int width_ = bottom[0]->width(); 
  int channels_ = bottom[0]->channels();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  unsigned* num_pos_data = num_pos_.mutable_cpu_data();
  Dtype* sum_data = sum_.mutable_cpu_data();
  Dtype* sum_sq_data = sum_sq_.mutable_cpu_data();
  Dtype* sum_prod_data = sum_prod_.mutable_cpu_data();
  unsigned* hist_data = hist_.mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    int offset = n * num_pos_.count();
    for (int i = 0; i < num_pos_.count(); ++i) {
      if (bottom_data[offset + i] > 0) {
        num_pos_data[i] ++;
      }
      sum_data[i] += bottom_data[offset + i];
      sum_sq_data[i] += bottom_data[offset + i] * bottom_data[offset + i];

      int c = i / (height_ * width_);
      int bin = std::floor(bottom_data[offset + i]) + hist_res;
      bin = std::max(bin, 0);
      bin = std::min(bin, (int)hist_res * 2);
      hist_data[bin * channels_ + c] ++;
    }
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_, height_ * width_,
        (Dtype)1., bottom_data + offset, bottom_data + offset,
        (Dtype)1., sum_prod_data);
    num_sample_ ++;
  }
}

template <typename Dtype>
void ReLUModLayer<Dtype>::PrintAnalysis() {
  if (sum_.height() == 1 && sum_.width() == 1) { // only for conv layers
    return;
  }
  FILE* pfile;
  string filename = this->layer_param_.name() + "-analysis";
  pfile = fopen(filename.c_str(), "w+");
  const unsigned* num_pos_data = num_pos_.cpu_data();
  const Dtype* sum_data = sum_.cpu_data();
  const Dtype* sum_sq_data = sum_sq_.cpu_data();
  const Dtype* sum_prod_data = sum_prod_.cpu_data();
  const unsigned* hist_data = hist_.cpu_data();
  for (int i = 0; i < num_pos_.count(); ++i) {
    // fprintf(pfile, "%f\n", double(num_pos_data[i]) / double(num_sample_));
    // fprintf(pfile, "%f\n", sum_data[i] / num_sample_); // print mean value
    // fprintf(pfile, "%f\n", sum_sq_data[i] / num_sample_ - sum_data[i] * sum_data[i] / num_sample_ / num_sample_); // print variance
  }

  // print hist
  for (int i = 0; i < hist_.count(); ++i) {
    fprintf(pfile, "%d\n", hist_data[i]);
  }
  // print conv matrix
  int channels_ = sum_.channels();
  int height_ = sum_.height();
  int width_ = sum_.width();
  vector<Dtype> mean_;
  for (int c = 0; c < channels_; ++c) {
    Dtype mean_channel = 0;
    for (int i = 0; i < height_ * width_; ++i) {
      mean_channel += sum_data[c * height_ * width_ + i];
      mean_channel /= num_sample_ * height_ * width_;
      mean_.push_back(mean_channel);
    }
  }
  /*
  for (int y = 0; y < channels_; ++y) {
    for (int x = 0; x < channels_; ++x) {
      fprintf(pfile, "%f ", sum_prod_data[y * channels_ + x] / num_sample_ / height_ / width_ - mean_[y] * mean_[x]);
    }
    fprintf(pfile, "\n");
  }
  */
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
