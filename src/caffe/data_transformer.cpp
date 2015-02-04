#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  // Bias value for color jittering
  if (phase_ == Caffe::TRAIN) {
    caffe_rng_gaussian(3, (Dtype)0., (Dtype)0.1, bias_rand_);
    bias_val_[0] = bias_pca_[0][0] * bias_rand_[0] + 
      bias_pca_[1][0] * bias_rand_[1] +
      bias_pca_[2][0] * bias_rand_[2];
    bias_val_[1] = bias_pca_[0][1] * bias_rand_[0] + 
      bias_pca_[1][1] * bias_rand_[1] +
      bias_pca_[2][1] * bias_rand_[2];
    bias_val_[2] = bias_pca_[0][2] * bias_rand_[0] + 
      bias_pca_[1][2] * bias_rand_[1] +
      bias_pca_[2][2] * bias_rand_[2];
  } else {
    bias_val_[0] = (Dtype)0.;
    bias_val_[1] = (Dtype)0.;
    bias_val_[2] = (Dtype)0.;
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index] + bias_val_[c]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index] + bias_val_[c]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        int c = j / (height * width);
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j] + bias_val_[c]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        int c = j / (height * width);
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j] + bias_val_[c]) * scale;
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
  InitBias();
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

template <typename Dtype>
void DataTransformer<Dtype>::InitBias() {
  bias_pca_[0][0] = (Dtype)66.3793;
  bias_pca_[0][1] = (Dtype)68.0404;
  bias_pca_[0][2] = (Dtype)67.8507;
  bias_pca_[1][0] = (Dtype)24.8792;
  bias_pca_[1][1] = (Dtype)-0.2223;
  bias_pca_[1][2] = (Dtype)-24.1168;
  bias_pca_[2][0] = (Dtype)-7.1852;
  bias_pca_[2][1] = (Dtype)14.5351;
  bias_pca_[2][2] = (Dtype)-7.5464;
}

INSTANTIATE_CLASS(DataTransformer);
}  // namespace caffe
