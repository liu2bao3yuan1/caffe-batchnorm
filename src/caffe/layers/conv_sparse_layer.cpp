#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSparseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(4);
    } else {
      this->blobs_.resize(3);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        1, 1, channels_, channels_ / group_));
    this->blobs_[1].reset(new Blob<Dtype>(
        1, channels_, kernel_h_ * kernel_w_, kernel_h_ * kernel_w_));
    this->blobs_[2].reset(new Blob<Dtype>(
        num_output_, channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    weight_filler->Fill(this->blobs_[1].get());
    weight_filler->Fill(this->blobs_[2].get());
    // If necessary, initialize and fill the biases:
    // 1 x 1 x 1 x output channels
    if (bias_term_) {
      this->blobs_[3].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[3].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // Set up ptrs for gemm_batch
  if (Caffe::mode() == Caffe::GPU) {
    cudaMalloc(&col_data_ptrs_gpu, channels_*sizeof(Dtype*));
    cudaMalloc(&col_diff_ptrs_gpu, channels_*sizeof(Dtype*));
    cudaMalloc(&w1_data_ptrs_gpu, channels_*sizeof(Dtype*));
    cudaMalloc(&w1_diff_ptrs_gpu, channels_*sizeof(Dtype*));
    cudaMalloc(&c1_data_ptrs_gpu, channels_*sizeof(Dtype*));
    cudaMalloc(&c1_diff_ptrs_gpu, channels_*sizeof(Dtype*));
  }
}

template <typename Dtype>
void ConvolutionSparseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  height_out_ =
      (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  for (int top_id = 0; top_id < top->size(); ++top_id) {
    (*top)[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  // Prepare the matrix multiplication computation.
  // Each input will be convolved as a single GEMM.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  N_ = height_out_ * width_out_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage.
  col_buffer_.Reshape(1, channels_ * kernel_h_ * kernel_w_, height_out_, width_out_);
  c0_buffer_.Reshape(1, channels_, height_, width_);
  c1_buffer_.Reshape(1, channels_ * kernel_h_ * kernel_w_, height_out_, width_out_);

  for (int top_id = 0; top_id < top->size(); ++top_id) {
    (*top)[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ConvolutionSparseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    int weight_offset = M_ * K_;  // number of filter parameters in a group
    int col_offset = K_ * N_;  // number of values in an input region / column
    int top_offset = M_ * N_;  // number of values in an output region / column
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      im2col_cpu(bottom_data + bottom[i]->offset(n), channels_, height_,
          width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
          col_data);
      // Take inner products for groups.
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
          (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // Add bias.
      if (bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
            bias_multiplier_.cpu_data(),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
  }
}

template <typename Dtype>
void ConvolutionSparseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->cpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_.cpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->cpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_cpu_data();
      Dtype* col_diff = col_buffer_.mutable_cpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->cpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_cpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                   width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
                   stride_h_, stride_w_, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
                col_data + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->cpu_data();
          }
          for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                (Dtype)1., weight + weight_offset * g,
                top_diff + top[i]->offset(n) + top_offset * g,
                (Dtype)0., col_diff + col_offset * g);
          }
          // col2im back to the data
          col2im_cpu(col_diff, channels_, height_, width_,
              kernel_h_, kernel_w_, pad_h_, pad_w_,
              stride_h_, stride_w_, bottom_diff + (*bottom)[i]->offset(n));
        }
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
    col_data_ptrs_cpu[i] = col_data + kernel_h_*kernel_w_*N_*i;
    col_diff_ptrs_cpu[i] = col_diff + kernel_h_*kernel_w_*N_*i;
    w1_data_ptrs_cpu[i] = w1_data + i*kernel_h_*kernel_w_*kernel_h_*kernel_w_;
    w1_diff_ptrs_cpu[i] = w1_diff + i*kernel_h_*kernel_w_*kernel_h_*kernel_w_;
    c1_data_ptrs_cpu[i] = c1_data + kernel_h_*kernel_w_*N_*i;
    c1_diff_ptrs_cpu[i] = c1_diff + kernel_h_*kernel_w_*N_*i;
  }

  cudaMemcpy(col_data_ptrs_gpu, col_data_ptrs_cpu,channels_*sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(col_diff_ptrs_gpu, col_diff_ptrs_cpu,channels_*sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(w1_data_ptrs_gpu, w1_data_ptrs_cpu,channels_*sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(w1_diff_ptrs_gpu, w1_diff_ptrs_cpu,channels_*sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(c1_data_ptrs_gpu, c1_data_ptrs_cpu,channels_*sizeof(Dtype*), cudaMemcpyHostToDevice);
  cudaMemcpy(c1_diff_ptrs_gpu, c1_diff_ptrs_cpu,channels_*sizeof(Dtype*), cudaMemcpyHostToDevice);

  delete[] col_data_ptrs_cpu;
  delete[] col_diff_ptrs_cpu;
  delete[] w1_data_ptrs_cpu;
  delete[] w1_diff_ptrs_cpu;
  delete[] c1_data_ptrs_cpu;
  delete[] c1_diff_ptrs_cpu;
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionSparseLayer);
#endif

INSTANTIATE_CLASS(ConvolutionSparseLayer);

}  // namespace caffe
