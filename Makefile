all:
	nvcc vgg.cu  -O3 -g -std=c++14 -arch=sm_70 -lcudnn_static -lculibos -lcublas