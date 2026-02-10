#include <random>
#include <cstddef>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <iomanip>

#include "cpu_impl.hxx"

#ifdef CUDA_ENABLED
#include <alpaka/alpaka.hpp>
#include "gpu_impl.hxx"
#endif

#ifndef CUDA_ENABLED
#include <alpaka/alpaka.hpp>
#endif

using Dim = alpaka::DimInt<2>;
using Idx = std::size_t;

std::vector<float> runMatMulCpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    size_t M, size_t N, size_t K);
    
void printMatrix(const std::vector<float>& mat, Idx rows, Idx cols, const std::string& name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):" << std::endl;
    for(Idx i = 0; i < rows; ++i) {
        for(Idx j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {

    constexpr Idx M = 4, K = 4, N = 4;
    
    // Use simple deterministic values instead of random for debugging
    std::vector<float> A(M*K);
    std::vector<float> B(K*N);
    
    // Fill with simple values: A = identity matrix, B = matrix with 1,2,3,4 in first row
    for(Idx i = 0; i < M; ++i) {
        for(Idx j = 0; j < K; ++j) {
            A[i * K + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    for(Idx i = 0; i < K; ++i) {
        for(Idx j = 0; j < N; ++j) {
            B[i * N + j] = static_cast<float>(j + 1);  // Column j gets value j+1
        }
    }
    
    // Print input matrices
    printMatrix(A, M, K, "Matrix A");
    printMatrix(B, K, N, "Matrix B");

    std::cout << "Running CPU matrix multiplication..." << std::endl;
    auto C_cpu = runMatMulCpu(A, B, M, N, K);
    std::cout << "CPU Result:" << std::endl;
    printMatrix(C_cpu, M, N, "Matrix C (CPU)");

#ifdef CUDA_ENABLED
    std::cout << "Running GPU matrix multiplication..." << std::endl;
    auto C_gpu = runMatMulGpu<alpaka::TagGpuCudaRt>(A, B, M, N, K);
    std::cout << "GPU Result:" << std::endl;
    printMatrix(C_gpu, M, N, "Matrix C (GPU)");
#else
    std::cout << "GPU support not enabled (compile with ENABLE_CUDA=ON)" << std::endl;
#endif
}