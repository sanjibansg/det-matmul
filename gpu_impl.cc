#include "kernel.hxx"

#include <random>
#include <cstddef>
#include <alpaka/alpaka.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <thread>
#include <atomic>
#include <iomanip>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template<typename TAcc>
std::vector<float> runMatMul(const std::vector<float>& A,
               const std::vector<float>& B,
               Idx M, Idx N, Idx K) {
    
    using Acc = alpaka::TagToAcc<TAcc, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<DevAcc, QueueProperty>;  
    using Vec = alpaka::Vec<Dim, Idx>;
    
    alpaka::Platform<Acc> const platform{};
    DevAcc devAcc = alpaka::getDevByIdx(platform, 0);
    alpaka::PlatformCpu platformHost{};
    alpaka::DevCpu hostAcc = alpaka::getDevByIdx(platformHost, 0);
    QueueAcc queue{devAcc};
    
    Vec const extentA(static_cast<Idx>(M), static_cast<Idx>(K));
    Vec const extentB(static_cast<Idx>(K), static_cast<Idx>(N));
    Vec const extentC(static_cast<Idx>(M), static_cast<Idx>(N));

    // Allocate buffers
    auto hostBuf_A = alpaka::allocBuf<float, Idx>(hostAcc, extentA);
    auto hostBuf_B = alpaka::allocBuf<float, Idx>(hostAcc, extentB);
    auto hostBuf_C = alpaka::allocBuf<float, Idx>(hostAcc, extentC);
    
    std::memcpy(alpaka::getPtrNative(hostBuf_A), A.data(), M*K*sizeof(float));
    std::memcpy(alpaka::getPtrNative(hostBuf_B), B.data(), K*N*sizeof(float));
    
    auto deviceBuf_A = alpaka::allocBuf<float, Idx>(devAcc, extentA);
    auto deviceBuf_B = alpaka::allocBuf<float, Idx>(devAcc, extentB);
    auto deviceBuf_C = alpaka::allocBuf<float, Idx>(devAcc, extentC);
    
    // Copy to device
    alpaka::memcpy(queue, deviceBuf_A, hostBuf_A);
    alpaka::memcpy(queue, deviceBuf_B, hostBuf_B);
    
    // Set up work division
    alpaka::KernelCfg<Acc> const kernelCfg
        = {extentC, Vec::ones(), false, alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};
    
    MatMulKernel kernel{};
    auto const workDiv = alpaka::getValidWorkDiv<Acc>(kernelCfg, devAcc, kernel,
         alpaka::getPtrNative(deviceBuf_A),
         alpaka::getPtrNative(deviceBuf_B),
         alpaka::getPtrNative(deviceBuf_C),
         M, N, K
    );

    std::cout << "Work division: gridExtent = [" 
              << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv)[0] << ", "
              << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv)[1] << "], "
              << "blockExtent = [" 
              << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(workDiv)[0] << ", "
              << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(workDiv)[1] << "]" << std::endl;

    // Execute kernel
    alpaka::exec<Acc>(
        queue,
        workDiv,
        kernel,
        alpaka::getPtrNative(deviceBuf_A),
        alpaka::getPtrNative(deviceBuf_B),
        alpaka::getPtrNative(deviceBuf_C),
        M, N, K
    );
    
    // Copy result back
    alpaka::memcpy(queue, hostBuf_C, deviceBuf_C);
    alpaka::wait(queue);
    
    std::vector<float> C(M*N);
    std::memcpy(C.data(), alpaka::getPtrNative(hostBuf_C), M*N*sizeof(float));
    return C;
}
#endif