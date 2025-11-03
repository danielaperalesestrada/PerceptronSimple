#ifndef PERCEPTRON_CUH
#define PERCEPTRON_CUH

__global__ void perceptron_forward_batch(
    const float* d_weights, const float* d_images, float* d_output,
    int n_pixels, int n_images, const float* bias);

__global__ void perceptron_update_batch(
    float* d_weights, const float* d_images, const float* d_output,
    const unsigned char* d_labels, int n_pixels, int n_images,
    int target_class, float eta, float* d_bias);

#endif
