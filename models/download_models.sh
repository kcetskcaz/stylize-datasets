#!/usr/bin/env bash
cd models
# The VGG-19 network is obtained by:
# 1. converting vgg_normalised.caffemodel to .t7 using loadcaffe
# 2. inserting a convolutional module at the beginning to preprocess the image
# 3. replacing zero-padding with reflection-padding
# The original vgg_normalised.caffemodel can be obtained with:
# "wget -c --no-check-certificate https://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel"
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU' -O vgg_normalised.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr' -O decoder.pth
cd ..
