#ifndef CAFFE_UTIL_EXP_H_
#define CAFFE_UTIL_EXP_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
void conv_neuron_stats(const Blob<Dtype>& weight_blob, const string& filename);

}  // namespace caffe

#endif  // CAFFE_UTIL_EXP_H_
