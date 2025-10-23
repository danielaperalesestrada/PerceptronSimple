#pragma once
#include <cuda_runtime.h>
#include <math.h>
__global__ void perceptron_forward(float* d_weights, float* d_images, float* d_output,int n_pixels, int n_images);
__global__ void perceptron_update(float* d_weights, float* d_images, float* d_output, unsigned char* d_labels, int n_pixels, int n_images, int target_class, float eta);
