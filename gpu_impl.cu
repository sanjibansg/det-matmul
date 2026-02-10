#include "gpu_impl.inl"
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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template std::vector<float> runMatMulGpu<alpaka::TagGpuCudaRt>(
    const std::vector<float>&,
    const std::vector<float>&,
    Idx, Idx, Idx);
#endif