#include "perceptron.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void perceptron_forward_batch(
    const float* d_weights, const float* d_images, float* d_output,
    int n_pixels, int n_images, float bias)
{
    int img_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (img_idx >= n_images) return;

    float sum = bias;
    for (int i = 0; i < n_pixels; ++i)
        sum += d_weights[i] * d_images[img_idx * n_pixels + i];

    d_output[img_idx] = 1.0f / (1.0f + expf(-sum));
}

__global__ void perceptron_update_batch(
    float* d_weights, const float* d_images, const float* d_output,
    const unsigned char* d_labels, int n_pixels, int n_images,
    int target_class, float eta, float* d_bias)
{
    extern __shared__ float shared_grad[]; // espacio compartido para gradientes
    int tid = threadIdx.x;

    // inicializar gradientes locales
    for (int i = tid; i < n_pixels; i += blockDim.x)
        shared_grad[i] = 0.0f;
    __syncthreads();

    // cada hilo procesa una imagen
    int img_idx = blockIdx.x * blockDim.x + tid;
    if (img_idx < n_images) {
        float t = (d_labels[img_idx] == target_class) ? 1.0f : 0.0f;
        float error = t - d_output[img_idx];

        for (int i = 0; i < n_pixels; ++i)
            shared_grad[i] += error * d_images[img_idx * n_pixels + i];

        atomicAdd(d_bias, eta * error);
    }
    __syncthreads();

    // reducción: aplicar gradiente acumulado al peso global
    for (int i = tid; i < n_pixels; i += blockDim.x)
        d_weights[i] += eta * shared_grad[i];
}
