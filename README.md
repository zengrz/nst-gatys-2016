# Neural Style Transfer

This project is a experimentation of the popular [Neural Style Transfer][1] by Gatys et al.

This script automatically explores the parameter space by randomly selecting pairs of images, learning rate, and relative importance of style and content images.

## Setup

1. In the root directory of this project, create folder `pretrained-model` and place the model you wish to use (in .mat format) in the folder. To download pretrained model, visit [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/) and download the [imagenet-vgg-verydeep-19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat).

2. In the root directory of this project, create folder `images` and place images of dimensions CONFIG.IMAGE_WIDTH X CONFIG.IMAGE_HEIGHT. See utils.py for configurations.

## Run

Navigate to `src` directory and execute `python main.py`.

## Reference
```
Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2414-2423).
```

[1]:https://arxiv.org/pdf/1508.06576.pdf
