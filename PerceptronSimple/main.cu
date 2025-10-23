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

int main() {
    // 1️⃣ Cargar un batch
    auto batch = load_cifar10_batch("D:/Usuarios/Descargas/data/data/data_batch_1.bin");

    // 2️⃣ Convertir imágenes a float
    std::vector<float> h_images(N_IMAGES * N_PIXELS);
    std::vector<unsigned char> h_labels(N_IMAGES);
    for (int i = 0; i < N_IMAGES; ++i) {
        h_labels[i] = batch[i].label;
        for (int j = 0; j < N_PIXELS; ++j)
            h_images[i * N_PIXELS + j] = batch[i].pixels[j] / 255.0f;
    }

    // 3️⃣ Inicializar pesos para 10 perceptrones
    std::vector<float> h_weights(N_CLASSES * N_PIXELS);
    for (auto& w : h_weights) w = ((float)rand() / RAND_MAX) * 0.01f;

    // 4️⃣ Copiar a GPU
    float* d_images, * d_weights, * d_output;
    unsigned char* d_labels;
    cudaMalloc(&d_images, h_images.size() * sizeof(float));
    cudaMalloc(&d_weights, h_weights.size() * sizeof(float));
    cudaMalloc(&d_output, N_IMAGES * sizeof(float));
    cudaMalloc(&d_labels, h_labels.size() * sizeof(unsigned char));

    cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 5️⃣ Entrenamiento simple (1 epoch por ahora)
    int threads = 256;
    int blocks = (N_IMAGES + threads - 1) / threads;

    for (int c = 0; c < N_CLASSES; ++c) {
        // forward
        perceptron_forward << <blocks, threads >> > (d_weights + c * N_PIXELS, d_images, d_output, N_PIXELS, N_IMAGES);
        cudaDeviceSynchronize();

        // update
        perceptron_update << <blocks, threads >> > (d_weights + c * N_PIXELS, d_images, d_output, d_labels, N_PIXELS, N_IMAGES, c, ETA);
        cudaDeviceSynchronize();
    }

    std::cout << "Epoch completada" << std::endl;
    cudaFree(d_images); cudaFree(d_weights); cudaFree(d_output); cudaFree(d_labels);
}