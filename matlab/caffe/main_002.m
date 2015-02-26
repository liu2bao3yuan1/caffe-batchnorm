clear; close all; clc;
im = imread('../../examples/images/cat.jpg');
use_gpu = 1;
if ~exist('weight_conv.mat', 'file')
  weight_conv = cell(0,0);
  for iter = 100:200:14000
    fprintf('iter = %d\n', iter);
    model_def_file = '../../models/symrelu/deploy.prototxt';
    model_file = sprintf('../../models/symrelu/caffenet_train_conv_iter_%d.caffemodel', iter);
  %   model_file = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';

    caffe('reset');
    matcaffe_init(use_gpu, model_def_file, model_file);
    weights = caffe('get_weights');
    weights = weights(1:5);
    weight_conv{numel(weight_conv)+1} = weights;
  end
  save('weight_conv.mat', 'weight_conv', '-v7.3');
else
  load('weight_conv.mat');
end

ndiff_all = [];
n_iter = numel(weight_conv);
step = 1;
conv = 4;
for iter = 1 + step : n_iter
%   w_ = weight_conv{iter - step}(conv).weights{1};
  w_ = weight_conv{n_iter}(conv).weights{1};
  w = weight_conv{iter}(conv).weights{1};
  for ifilter = 1:size(w,4)
    w_(:,:,:,ifilter) = w_(:,:,:,ifilter) / norm(reshape(w_(:,:,:,ifilter),1,[]));
    w(:,:,:,ifilter) = w(:,:,:,ifilter) / norm(reshape(w(:,:,:,ifilter),1,[]));
  end
%   w_ = w_ / norm(w_(:));
%   w = w / norm(w(:));
  ndiff = norm(w(:) - w_(:)) / norm(w(:));
  fprintf('iter = %d, normalized diff = %f\n', iter, ndiff);
  ndiff_all = [ndiff_all, ndiff];
end
plot(ndiff_all);


return;


%% conv1
for iter = 1:numel(weight_conv)
  fprintf('iter = %d\n', iter);
  w = weight_conv{iter}(1).weights{1};
  w = permute(w, [1,4,2,3]);
  w = reshape(w, [11*8,12,11,3]);
  w = permute(w, [3,2,1,4]);
  w = reshape(w, [11*12,11*8,3]);
  w = permute(w, [2,1,3]);
  w = w / mean(abs(w(:))) / 4 + 0.5;
  imshow(w);
  drawnow;
  pause(0.1);
end

