#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/exp.hpp"
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;

namespace caffe {

template <typename Dtype>
void conv_neuron_stats(const Blob<Dtype>& weight_blob, const string& filename) {
  // const Dtype* weight_data = weight_blob.cpu_data();
  const Dtype* weight_diff = weight_blob.cpu_diff();
  int step = weight_blob.channels() * weight_blob.height() * weight_blob.width();
  vector<Dtype> norm_diff_vec;
  for (int n = 0; n < weight_blob.num(); ++n) {
    Dtype norm_diff = (Dtype)0;
    for (int i = 0; i < step; ++i) {
      norm_diff += weight_diff[n * step + i] * weight_diff[n * step + i]; 
    }
    norm_diff = std::sqrt(norm_diff);
    norm_diff_vec.push_back(norm_diff);
  }
  sort(norm_diff_vec.begin(), norm_diff_vec.end(),  greater<Dtype>());
  ofstream file;
  file.open(filename.c_str(), ios::out | ios::app);
  for (int i = 0; i < norm_diff_vec.size(); ++i) {
    file << norm_diff_vec[i] << " ";
  }
  file << endl;
  file.close();
}

template
void conv_neuron_stats<float>(const Blob<float>& weight_blob, const string& filename);
template
void conv_neuron_stats<double>(const Blob<double>& weight_blob, const string& filename);

}  // namespace caffe
