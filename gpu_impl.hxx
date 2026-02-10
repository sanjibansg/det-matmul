#pragma once

#include <vector>
#include <cstddef>

using Idx = std::size_t;

template<typename TAccTag>
std::vector<float> runMatMulGpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    Idx M, Idx N, Idx K);