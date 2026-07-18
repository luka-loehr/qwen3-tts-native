cmake_minimum_required(VERSION 3.24)

project(qwen3_tts_cuda VERSION 0.2.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit 13.0 REQUIRED)

add_library(qwen3_tts_cuda SHARED
    src/gemv.cu
    src/primitive_parity.cu
    src/runtime.cu
    src/sampling_kernels.cu
    src/talker.cu
    src/talker_kernels.cu
    src/weights.cu
)

target_include_directories(qwen3_tts_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(qwen3_tts_cuda PRIVATE cxx_std_17)
set_target_properties(qwen3_tts_cuda PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_ARCHITECTURES 121-real
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE YES
    VISIBILITY_INLINES_HIDDEN YES
)

target_compile_options(qwen3_tts_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)

target_link_libraries(qwen3_tts_cuda PRIVATE CUDA::cublas CUDA::cudart)
