#include "data_loader.h"
#include "perceptron.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib>

#define N_PIXELS 3072
#define N_CLASSES 10
#define N_IMAGES 10000
#define ETA 0.01f
#define BATCH_SIZE 256
#define N_EPOCHS 50

const char* CLASS_NAMES[10] = {
    "avion", "auto", "pajaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camion"
};

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
        << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(1); \
    }

int predict_image(const std::vector<float>&weights,
    const std::vector<float>&bias,
    const std::vector<float>&image)
{
    float max_score = -1e9;
    int pred = -1;

    for (int c = 0; c < N_CLASSES; ++c) {
        float sum = bias[c];
        for (int j = 0; j < N_PIXELS; ++j)
            sum += weights[c * N_PIXELS + j] * image[j];
        if (sum > max_score) {
            max_score = sum;
            pred = c;
        }
    }
    return pred;
}

int main() {
    auto batch = load_cifar10_batch("D:/Usuarios/Descargas/data/data/data_batch_1.bin");

    std::vector<float> h_images(N_IMAGES * N_PIXELS);
    std::vector<unsigned char> h_labels(N_IMAGES);
    for (int i = 0; i < N_IMAGES; ++i) {
        h_labels[i] = batch[i].label;
        for (int j = 0; j < N_PIXELS; ++j)
            h_images[i * N_PIXELS + j] = batch[i].pixels[j] / 255.0f;
    }

    std::vector<float> h_weights(N_CLASSES * N_PIXELS);
    std::vector<float> h_bias(N_CLASSES, 0.0f);
    for (auto& w : h_weights) w = ((float)rand() / RAND_MAX) * 0.01f;

    float* d_images, * d_weights, * d_output, * d_bias;
    unsigned char* d_labels;
    CUDA_CHECK(cudaMalloc(&d_images, h_images.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, h_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N_IMAGES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, h_labels.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = BATCH_SIZE;
    int blocks = (N_IMAGES + threads - 1) / threads;
    size_t sharedMem = N_PIXELS * sizeof(float);

    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        for (int c = 0; c < N_CLASSES; ++c) {
            float* w_c = d_weights + c * N_PIXELS;
            float* b_c = d_bias + c;

            perceptron_forward_batch << <blocks, threads >> > (w_c, d_images, d_output, N_PIXELS, N_IMAGES, h_bias[c]);
            CUDA_CHECK(cudaDeviceSynchronize());

            perceptron_update_batch << <blocks, threads, sharedMem >> > (w_c, d_images, d_output, d_labels,
                N_PIXELS, N_IMAGES, c, ETA, b_c);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::cout << "Epoch " << epoch + 1 << " completada" << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights, h_weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias.data(), d_bias, h_bias.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int correct = 0;
    for (int i = 0; i < N_IMAGES; ++i) {
        float max_score = -1e9;
        int pred = -1;
        for (int c = 0; c < N_CLASSES; ++c) {
            float sum = h_bias[c];
            for (int j = 0; j < N_PIXELS; ++j)
                sum += h_weights[c * N_PIXELS + j] * h_images[i * N_PIXELS + j];
            if (sum > max_score) { max_score = sum; pred = c; }
        }
        if (pred == h_labels[i]) correct++;
    }
    std::cout << "Precision final: " << (100.0f * correct / N_IMAGES) << "%" << std::endl;

    std::cout << "\n--- Prueba con una imagen ---\n";

    // Tomemos una imagen del batch (ejemplo: la número 5)
    int test_idx = 5;
    std::vector<float> test_img(N_PIXELS);
    for (int j = 0; j < N_PIXELS; ++j)
        test_img[j] = h_images[test_idx * N_PIXELS + j];

    // Predicción
    int pred = predict_image(h_weights, h_bias, test_img);

    std::cout << "Etiqueta real: " << (int)h_labels[test_idx]
        << " (" << CLASS_NAMES[h_labels[test_idx]] << ")\n";
    std::cout << "Prediccion:   " << pred
        << " (" << CLASS_NAMES[pred] << ")\n";


    cudaFree(d_images); cudaFree(d_weights); cudaFree(d_output); cudaFree(d_labels); cudaFree(d_bias);
}
