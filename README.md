# The Breast Cancer Screening Framework
A computer-aided diagnosis system for the classification of mammography mass lesions

# Prerequisites

- A Cuda enabled GPU (The implementation was tested on Ubuntu mate 16.04.1 LTS 64-bit and Nvidia GeForce GTX 980M graphic card)
- Cuda 8.0 and CuDNN v5
- Python  2.7-3.5
- Opencv 3.1.0
- Theano 0.8.2
- Keras from https://github.com/fchollet/keras


To run the framework 

```sh
$ python BCSF.py
```
To compute the ROC curve and AUC for the independent dataset

```sh
$ python roc.py data
