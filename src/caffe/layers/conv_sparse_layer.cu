#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/// @brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void ConvolutionSparseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    Dtype* c0_data = c0_buffer_.mutable_gpu_data();
    Dtype* c1_data = c1_buffer_.mutable_gpu_data();

    const Dtype* w0_data = this->blobs_[0]->gpu_data();
    const Dtype* w1_data = this->blobs_[1]->gpu_data();
    const Dtype* w2_data = this->blobs_[2]->gpu_data();

    int bottom_offset = height_ * width_ * channels_ / group_;
    int c0_offset = height_ * width_ * channels_ / group_;
    int w0_offset = channels_ * channels_ / group_ / group_;
    // int col_offset = K_ * N_;
    int c1_offset = K_ * N_;
    int w2_offset = M_ * K_;
    int top_offset = M_ * N_;

    UpdatePtrs();

    for (int n = 0; n < num_; ++n) {
      // PCA on channels
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ / group_, height_ * width_, channels_ / group_,
            (Dtype)1., w0_data + g * w0_offset, bottom_data + bottom[i]->offset(n) + g * bottom_offset,
            (Dtype)0., c0_data + g * c0_offset); 
      }
      // im2col
      im2col_gpu(c0_data, channels_, height_,
          width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_data);
      // multiply with w1, save to c1 
      caffe_gpu_gemm_batch<Dtype>(CblasNoTrans, CblasNoTrans,
          kernel_h_ * kernel_w_, N_, kernel_h_ * kernel_w_,
          (Dtype)1., (const Dtype**)w1_data_ptrs_gpu, (const Dtype**)col_data_ptrs_gpu,
          (Dtype)0., c1_data_ptrs_gpu, channels_);
      for (int g = 0; g < group_; ++g) {
        // multiply with w2, save to top 
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            (Dtype)1., w2_data + w2_offset * g, c1_data + c1_offset * g,
            (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // add bias
      bias_multiplier_.gpu_data();
      if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[3]->gpu_data(),
            bias_multiplier_.gpu_data(),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
  }
}

/// @brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void ConvolutionSparseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype* w0_data = this->blobs_[0]->mutable_gpu_data();
  Dtype* w0_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* w1_data = this->blobs_[1]->mutable_gpu_data();
  Dtype* w1_diff = this->blobs_[1]->mutable_gpu_diff();
  Dtype* w2_data = this->blobs_[2]->mutable_gpu_data();
  Dtype* w2_diff = this->blobs_[2]->mutable_gpu_diff();
  Dtype* bias_diff = this->blobs_[3]->mutable_gpu_diff();

  // zero accumulated gradients
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), w0_diff);
  caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), w1_diff);
  caffe_gpu_set(this->blobs_[2]->count(), Dtype(0), w2_diff);
  if (bias_term_) {
    caffe_gpu_set(this->blobs_[3]->count(), Dtype(0), bias_diff);
  }

  int bottom_offset = height_ * width_ * channels_ / group_;
  int c0_offset = height_ * width_ * channels_ / group_;
  int w0_offset = channels_ * channels_ / group_ / group_;
  int c1_offset = K_ * N_;
  int w2_offset = M_ * K_;
  int top_offset = M_ * N_;

  for (int i = 0; i < top.size(); ++i) {
    Dtype* bottom_data = (*bottom)[i]->mutable_gpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
    Dtype* top_data = top[i]->mutable_gpu_data();
    Dtype* top_diff = top[i]->mutable_gpu_diff();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    Dtype* col_diff = col_buffer_.mutable_gpu_diff();
    Dtype* c0_data = c0_buffer_.mutable_gpu_data();
    Dtype* c0_diff = c0_buffer_.mutable_gpu_diff();
    Dtype* c1_data = c1_buffer_.mutable_gpu_data();
    Dtype* c1_diff = c1_buffer_.mutable_gpu_diff();

    UpdatePtrs();

    for (int n = 0; n < num_; ++n) {
      // Since we saved memory in the forward pass by not storing all col
      // data, we will need to recompute them.
      /******************  Forward Pass  *******************/
      // PCA on channels
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ / group_, height_ * width_, channels_ / group_,
            (Dtype)1., w0_data + g * w0_offset, bottom_data + (*bottom)[i]->offset(n) + g * bottom_offset,
            (Dtype)0., c0_data + g * c0_offset); 
      }
      // im2col
      im2col_gpu(c0_data, channels_, height_,
          width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_data);
      // multiply with w1, save to c1 
      caffe_gpu_gemm_batch<Dtype>(CblasNoTrans, CblasNoTrans,
          kernel_h_ * kernel_w_, N_, kernel_h_ * kernel_w_,
          (Dtype)1., (const Dtype**)w1_data_ptrs_gpu, (const Dtype**)col_data_ptrs_gpu,
          (Dtype)0., c1_data_ptrs_gpu, channels_);
      for (int g = 0; g < group_; ++g) {
        // multiply with w2, save to top 
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            (Dtype)1., w2_data + w2_offset * g, c1_data + c1_offset * g,
            (Dtype)0., top_data + top[i]->offset(n) + top_offset * g);
      }
      // add bias
      if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[3]->gpu_data(),
            bias_multiplier_.gpu_data(),
            (Dtype)1., top_data + top[i]->offset(n));
      }

      /******************  Backward Pass  *******************/
      // bias
      if (bias_term_) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            (Dtype)1., top_diff + top[i]->offset(n),
            bias_multiplier_.gpu_data(), (Dtype)1.,
            bias_diff);
      }
      for (int g = 0; g < group_; ++g) {
        // w2_diff
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
            (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g, c1_data + c1_offset * g,
            (Dtype)1., w2_diff + w2_offset * g);
        // c1_diff
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
            (Dtype)1., w2_data + w2_offset * g, top_diff + top[i]->offset(n) + top_offset * g, 
            (Dtype)0., c1_diff + c1_offset * g);
      }
      // w1_diff
      caffe_gpu_gemm_batch<Dtype>(CblasNoTrans, CblasTrans, 
          kernel_h_ * kernel_w_, kernel_h_ * kernel_w_, N_,
          (Dtype)1., (const Dtype**)c1_diff_ptrs_gpu, (const Dtype**)col_data_ptrs_gpu, 
          (Dtype)1., w1_diff_ptrs_gpu, channels_);
      // col_diff
      caffe_gpu_gemm_batch<Dtype>(CblasTrans, CblasNoTrans, 
          kernel_h_ * kernel_w_, N_, kernel_h_ * kernel_w_,
          (Dtype)1., (const Dtype**)w1_data_ptrs_gpu, (const Dtype**)c1_diff_ptrs_gpu,
          (Dtype)0., col_diff_ptrs_gpu, channels_);
      // c0_diff
      col2im_gpu(col_diff, channels_, height_, width_,
          kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
          c0_diff);
      // w0
      for (int g = 0; g < group_; ++g) {
        // w0_diff
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_ / group_, channels_ / group_, height_ * width_, 
            (Dtype)1., c0_diff + g * c0_offset, bottom_data + (*bottom)[i]->offset(n) + g * bottom_offset,
            (Dtype)1., w0_diff + g * w0_offset);
        // bottom_diff
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_ / group_, height_ * width_, channels_ / group_,
            (Dtype)1., w0_data + g * w0_offset, c0_diff + g * c0_offset,
            (Dtype)0., bottom_diff + (*bottom)[i]->offset(n) + g * bottom_offset);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionSparseLayer<Dtype>::UpdatePtrs() {
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  Dtype* w1_data = this->blobs_[1]->mutable_gpu_data();
  Dtype* w1_diff = this->blobs_[1]->mutable_gpu_diff();
  Dtype* c1_data = c1_buffer_.mutable_gpu_data();
  Dtype* c1_diff = c1_buffer_.mutable_gpu_diff();

  // Setup gpu pointers for gemm batch mode 
  Dtype** col_data_ptrs_cpu = new Dtype*[channels_];
  Dtype** col_diff_ptrs_cpu = new Dtype*[channels_];
  Dtype** w1_data_ptrs_cpu = new Dtype*[channels_];
  Dtype** w1_diff_ptrs_cpu = new Dtype*[channels_];
  Dtype** c1_data_ptrs_cpu = new Dtype*[channels_];
  Dtype** c1_diff_ptrs_cpu = new Dtype*[channels_];

  for(int i=0; i<channels_; i++) {
    col_data_ptrs_cpu[i] = col_data + kernel_h_ * kernel_w_ * N_ * i;
    col_diff_ptrs_cpu[i] = col_diff + kernel_h_ * kernel_w_ * N_*i;
    w1_data_ptrs_cpu[i] = w1_data + i * kernel_h_ * kernel_w_ * kernel_h_ * kernel_w_;
    w1_diff_ptrs_cpu[i] = w1_diff + i * kernel_h_ * kernel_w_ * kernel_h_ * kernel_w_;
    c1_data_ptrs_cpu[i] = c1_data + kernel_h_ * kernel_w_ * N_ * i;
    c1_diff_ptrs_cpu[i] = c1_diff + kernel_h_ * kernel_w_ * N_ * i;
  }

  cudaMemcpy(col_data_ptrs_gpu, col_data_ptrs_cpu, channels_ * sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(col_diff_ptrs_gpu, col_diff_ptrs_cpu, channels_ * sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(w1_data_ptrs_gpu, w1_data_ptrs_cpu, channels_ * sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(w1_diff_ptrs_gpu, w1_diff_ptrs_cpu, channels_ * sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(c1_data_ptrs_gpu, c1_data_ptrs_cpu, channels_ * sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(c1_diff_ptrs_gpu, c1_diff_ptrs_cpu, channels_ * sizeof(Dtype*), cudaMemcpyHostToDevice);

  delete[] col_data_ptrs_cpu;
  delete[] col_diff_ptrs_cpu;
  delete[] w1_data_ptrs_cpu;
  delete[] w1_diff_ptrs_cpu;
  delete[] c1_data_ptrs_cpu;
  delete[] c1_diff_ptrs_cpu;
}

INSTANTIATE_CLASS(ConvolutionSparseLayer);

}  // namespace caffe

/*
template <typename Dtype>
void ConvolutionSparseLayer<Dtype>::Backward_gpu_org(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->gpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_.gpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->gpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_gpu_data();
      Dtype* col_diff = col_buffer_.mutable_gpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                   width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
                   stride_h_, stride_w_, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
                col_data + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                (Dtype)1., weight + weight_offset * g,
                top_diff + top[i]->offset(n) + top_offset * g,
                (Dtype)0., col_diff + col_offset * g);
          }
          // col2im back to the data
          col2im_gpu(col_diff, channels_, height_, width_,
              kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
              bottom_diff + (*bottom)[i]->offset(n));
        }
      }
    }
  }
}
*/
