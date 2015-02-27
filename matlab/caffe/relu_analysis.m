clear; close all; clc;
ksize = 13;
for layer = 3
  filename = sprintf('%s/relu-analysis/relu%d-analysis', pwd, layer);
  relu = dlmread(filename);
  relu = reshape(relu, ksize, ksize, []);
  nc = size(relu, 3);
  for c = 1:nc
    img = imresize(relu(:,:,c), 10, 'nearest');
    imshow(img);
    drawnow;
    keyboard;
  end
end