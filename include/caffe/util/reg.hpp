#ifndef CAFFE_UTIL_REG_H_
#define CAFFE_UTIL_REG_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
void caffe_l1_reg(const int n, const Dtype decay, const Dtype* val, Dtype* diff);
template <typename Dtype>
void caffe_l1_reg_group(const Dtype lambda, const int GROUP_, Blob<Dtype>& val_blob, Blob<Dtype>& diff_blob);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
