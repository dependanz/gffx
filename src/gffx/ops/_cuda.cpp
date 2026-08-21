/**
 * Playing around with a pybind11 binding
 * to a CUDA kernel.
 */

#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" int add_vectors_cuda(const float*, const float*, float*, int);

static pybind11::array_t<float> as_f32_1d(pybind11::array input, const char* name) {
    if (input.dtype().kind() != 'f' || input.dtype().itemsize() != 4)
        throw std::runtime_error(std::string(name) + " must be float32");
    if (input.ndim() != 1)
        throw std::runtime_error(std::string(name) + " must be 1-dimensional");
    
        return pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>(input);
}

pybind11::array_t<float> add_vectors(pybind11::array arr1, pybind11::array arr2) {
    pybind11::array_t<float> a = as_f32_1d(arr1, "a");
    pybind11::array_t<float> b = as_f32_1d(arr2, "b");
    if (a.size() != b.size())
        throw std::runtime_error("Input arrays must have the same size");
    
    pybind11::array_t<float> out(a.size());
    if (add_vectors_cuda(a.data(), b.data(), out.mutable_data(), (int) a.size()))
        throw std::runtime_error("CUDA kernel failed");
    
        return out;
}

PYBIND11_MODULE(_cuda, m) {
    m.def(
        "add_vectors", 
        &add_vectors, 
        "Add two float32 NumPy vectors (CUDA)"
    );
}