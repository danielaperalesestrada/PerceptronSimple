#include "perceptron.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void perceptron_forward_batch(
    const float* d_weights,
    const float* d_images,
    float* d_output,
    int n_pixels,
    int n_images,
    const float* d_bias)
{
    int img_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (img_idx >= n_images) return;

    float sum = *d_bias;

    const float* img = &d_images[img_idx * n_pixels];

    for (int i = 0; i < n_pixels; ++i)
        sum += d_weights[i] * img[i];

    d_output[img_idx] = sum;
}

__global__ void perceptron_update_batch(
    float* d_weights,
    const float* d_images,
    const float* d_output,
    const unsigned char* d_labels,
    int n_pixels, int n_images,
    int target_class,
    float eta,
    float* d_bias)
{
    int img_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (img_idx >= n_images) return;

    int y_true = (d_labels[img_idx] == target_class) ? 1 : -1;

    float output = d_output[img_idx];

    if (y_true * output > 0) return;

    const float* img = &d_images[img_idx * n_pixels];

    for (int i = 0; i < n_pixels; ++i)
        atomicAdd(&d_weights[i], eta * y_true * img[i]);

    atomicAdd(d_bias, eta * (float)y_true);
}
