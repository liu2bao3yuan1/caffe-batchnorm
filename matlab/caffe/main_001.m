clear; close all; clc;
im = imread('../../examples/images/cat.jpg');
use_gpu = 1;
model_def_file = '../../models/bvlc_reference_caffenet/deploy.prototxt';
model_file = '../../models/bvlc_reference_caffenet/caffenet_train_conv_iter_160.caffemodel';
matcaffe_init(use_gpu, model_def_file, model_file);
weights = caffe('get_weights');

