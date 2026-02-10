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
    
void printMatrix(const std::vector<float>& mat, Idx rows, Idx cols, const std::string& name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):" << std::endl;
    for(Idx i = 0; i < rows; ++i) {
        for(Idx j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void compareResults(const std::vector<float>& C_cpu, const std::vector<float>& C_gpu, 
                    Idx M, Idx N) {
    std::cout << "\n=== Comparing CPU vs GPU Results ===" << std::endl;
    
    if(C_cpu.size() != C_gpu.size()) {
        std::cout << "ERROR: Size mismatch! CPU size: " << C_cpu.size() 
                  << ", GPU size: " << C_gpu.size() << std::endl;
        return;
    }
    
    bool identical = true;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    Idx num_differences = 0;
    Idx max_diff_idx = 0;
    
    for(Idx i = 0; i < M * N; ++i) {
        float abs_diff = std::abs(C_cpu[i] - C_gpu[i]);
        
        // Check for bit-exact equality
        if(C_cpu[i] != C_gpu[i]) {
            identical = false;
            num_differences++;
            
            // Track maximum absolute difference
            if(abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
                max_diff_idx = i;
            }
            
            // Calculate relative difference (avoid division by zero)
            float denominator = std::max(std::abs(C_cpu[i]), 1e-10f);
            float rel_diff = abs_diff / denominator;
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
    }
    
    if(identical) {
        std::cout << "✓ PASS: Results are BIT-EXACT identical!" << std::endl;
        std::cout << "  All " << M * N << " elements match exactly." << std::endl;
    } else {
        std::cout << "✗ FAIL: Results differ!" << std::endl;
        std::cout << "  Number of differences: " << num_differences << " / " << M * N 
                  << " (" << std::fixed << std::setprecision(2) 
                  << (100.0 * num_differences / (M * N)) << "%)" << std::endl;
        std::cout << "  Maximum absolute difference: " << std::scientific << std::setprecision(6) 
                  << max_abs_diff << std::endl;
        std::cout << "  Maximum relative difference: " << std::scientific << std::setprecision(6) 
                  << max_rel_diff << std::endl;
        
        Idx row = max_diff_idx / N;
        Idx col = max_diff_idx % N;
        std::cout << "  Largest difference at position [" << row << ", " << col << "]:" << std::endl;
        std::cout << "    CPU value: " << std::fixed << std::setprecision(10) 
                  << C_cpu[max_diff_idx] << std::endl;
        std::cout << "    GPU value: " << std::fixed << std::setprecision(10) 
                  << C_gpu[max_diff_idx] << std::endl;
        std::cout << "    Difference: " << std::scientific << std::setprecision(6) 
                  << (C_gpu[max_diff_idx] - C_cpu[max_diff_idx]) << std::endl;
    }
    std::cout << "=====================================" << std::endl;
}

int main() {

    constexpr Idx M = 4, K = 4, N = 4;
    
    std::vector<float> A(M*K);
    std::vector<float> B(K*N);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);
    
    for(Idx i = 0; i < M; ++i) {
        for(Idx j = 0; j < K; ++j) {
            A[i * K + j] = dis(gen);
        }
    }
    
    for(Idx i = 0; i < K; ++i) {
        for(Idx j = 0; j < N; ++j) {
            B[i * N + j] = dis(gen);
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
    compareResults(C_cpu, C_gpu, M, N);
#else
    std::cout << "GPU support not enabled (compile with ENABLE_CUDA=ON)" << std::endl;
#endif
}