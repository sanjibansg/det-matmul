#include <alpaka/alpaka.hpp>

#include "kernel.hxx"
#include <random>
#include <cstddef>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <iomanip>


std::vector<float> runMatMulCpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    Idx M, Idx N, Idx K
)
{
    using Host = alpaka::AccCpuSerial<Dim, Idx>;
    using HostQueueProperty = alpaka::Blocking;
    using HostQueue = alpaka::Queue<Host, HostQueueProperty>;
    using Vec = alpaka::Vec<Dim, Idx>;

    alpaka::PlatformCpu platform{};
    auto hostAcc = alpaka::getDevByIdx(platform, 0);
    HostQueue queue{hostAcc};

    auto hostBuf_A = alpaka::allocBuf<float, Idx>(hostAcc, M * K);
    auto hostBuf_B = alpaka::allocBuf<float, Idx>(hostAcc, K * N);
    auto hostBuf_C = alpaka::allocBuf<float, Idx>(hostAcc, M * N);

    std::memcpy(alpaka::getPtrNative(hostBuf_A), A.data(), M*K*sizeof(float));
    std::memcpy(alpaka::getPtrNative(hostBuf_B), B.data(), K*N*sizeof(float));

    alpaka::wait(queue);

    MatMulKernel kernel{};

    Vec globalThreadExtent{M, N};
    Vec elementsPerThread{1, 1};

    alpaka::KernelCfg<Host> const hostKernelCfg{
        globalThreadExtent,
        elementsPerThread,
    };

    auto hostWorkDiv = alpaka::getValidWorkDiv(hostKernelCfg, hostAcc, kernel,
        alpaka::getPtrNative(hostBuf_A),
        alpaka::getPtrNative(hostBuf_B),
        alpaka::getPtrNative(hostBuf_C),
        M, N, K
    ); 
    

    alpaka::exec<Host>(
        queue,
        hostWorkDiv,
        kernel,
        alpaka::getPtrNative(hostBuf_A),
        alpaka::getPtrNative(hostBuf_B),
        alpaka::getPtrNative(hostBuf_C),
        M, N, K
    );

    alpaka::wait(queue);

    std::vector<float> C(M*N);
    std::memcpy(C.data(), alpaka::getPtrNative(hostBuf_C), M*N*sizeof(float));
    return C;
}