cmake_minimum_required(VERSION 3.26)
project(gffx LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit)

pybind11_add_module(_cpu_stub
    src/gffx/ops/_cpu_stub.cpp
)

set_target_properties(_cpu_stub PROPERTIES PREFIX "")
install(TARGETS _cpu_stub DESTINATION gffx/ops)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found, building CUDA extension")
    pybind11_add_module(_cuda
        src/gffx/ops/_cuda.cpp
        src/gffx/ops/_kernel.cu
    )

    set_target_properties(_cuda PROPERTIES PREFIX "")
    target_link_libraries(_cuda PRIVATE CUDA::cudart)
    install(TARGETS _cuda DESTINATION gffx/ops)

else()
    message(STATUS "CUDA not found, skipping CUDA extension build")
endif()