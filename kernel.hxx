#pragma once

#include <random>
#include <cstddef>
#include <alpaka/alpaka.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <iomanip>

using Dim = alpaka::DimInt<2>;
using Idx = std::size_t;

// ---------------- Kernel ----------------
struct MatMulKernel {
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const & acc,
        const T* A,
        const T* B,
        T* C,
        Idx M,
        Idx N,
        Idx K
    ) const {
        // Get indices
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        
        auto const i = globalThreadIdx[0];
        auto const j = globalThreadIdx[1];
        
        if(i < M && j < N) {
            T sum = static_cast<T>(0.0);
            for(Idx k = 0; k < K; ++k) {
                T a_val = A[i * K + k];
                T b_val = B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] = sum;
        }
    }
};