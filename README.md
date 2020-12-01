# Simple VGG16 in CUDA

This is a simple VGG16 net implemented in CUDA.

Simply run `make; ./a.out` to test the code. You may need to setup the environment for CUDA, cudnn, cublas.

The convolution layers use cudnn, and the FC layer uses cublas.
