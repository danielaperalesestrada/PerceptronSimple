#include "perceptron.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void perceptron_forward(
    float* d_weights, float* d_images, float* d_output,
    int n_pixels, int n_images)
{
    int img_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (img_idx >= n_images) return;

    float sum = 0.0f;
    for (int i = 0; i < n_pixels; ++i)
        sum += d_weights[i] * d_images[img_idx * n_pixels + i];

    d_output[img_idx] = 1.0f / (1.0f + expf(-sum)); // sigmoid
}

// kernel para actualizar pesos
__global__ void perceptron_update(
    float* d_weights, float* d_images, float* d_output,
    unsigned char* d_labels, int n_pixels, int n_images,
    int target_class, float eta)
{
    int img_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (img_idx >= n_images) return;

    float t = (d_labels[img_idx] == target_class) ? 1.0f : 0.0f;
    float error = t - d_output[img_idx];

    for (int i = 0; i < n_pixels; ++i)
        atomicAdd(&d_weights[i], eta * error * d_images[img_idx * n_pixels + i]);
}