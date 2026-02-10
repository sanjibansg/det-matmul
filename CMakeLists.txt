cmake_minimum_required(VERSION 3.21)
project(DeterministicMatMul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- Options ----------------
option(ENABLE_CUDA "Enable CUDA support" ON)

# ---------------- Paths ----------------
set(ALPAKA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/alpaka" CACHE PATH "Alpaka base path")

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_BASE "/usr/local/cuda" CACHE PATH "CUDA base path")
endif()

# ---------------- Sources ----------------
set(CXX_SOURCES
    main.cc
    cpu_impl.hxx
    kernel.hxx
    gpu_impl.hxx
    gpu_impl.inl
)

set(CUDA_SOURCES)
if(ENABLE_CUDA)
    list(APPEND CUDA_SOURCES gpu_impl.cu)
endif()

add_executable(DetMatMul ${CXX_SOURCES} ${CUDA_SOURCES})

# Define CPU accelerator for C++ files
target_compile_definitions(DetMatMul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED>
)

# Define CUDA accelerator and custom macro
if(ENABLE_CUDA)
    # For CUDA files only - define ALPAKA_ACC_GPU_CUDA_ENABLED
    target_compile_definitions(DetMatMul PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:ALPAKA_ACC_GPU_CUDA_ENABLED>
    )
    
    # For ALL files - define custom CUDA_ENABLED macro
    target_compile_definitions(DetMatMul PRIVATE
        CUDA_ENABLED
    )
endif()

target_include_directories(DetMatMul PRIVATE
    ${ALPAKA_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_CUDA)
    target_include_directories(DetMatMul PRIVATE
        ${CUDA_BASE}/include
    )
    
    target_link_directories(DetMatMul PRIVATE
        ${CUDA_BASE}/lib64
    )
endif()

# Compiler flags
target_compile_options(DetMatMul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -fno-fast-math
        -ffp-contract=off
    >
)

if(ENABLE_CUDA)
    target_compile_options(DetMatMul PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --fmad=false
            -Xcompiler=-fno-fast-math
        >
    )
    
    set_target_properties(DetMatMul PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    target_link_libraries(DetMatMul PRIVATE nvidia-ml)
endif()