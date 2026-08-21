#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

pybind11::array_t<float> add_vectors(pybind11::array arr1, pybind11::array arr2) {
    pybind11::array_t<float> a = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>(arr1);
    pybind11::array_t<float> b = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>(arr2);
    if (a.ndim() != 1 || b.ndim() != 1 || a.size() != b.size())
        throw std::runtime_error("Input arrays must be 1-dimensional and have the same size");

    pybind11::array_t<float> out(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        out.mutable_at(i) = a.at(i) + b.at(i);
    }

    return out;
}

PYBIND11_MODULE(_cpu_stub, m) {
    m.def(
        "add_vectors", 
        &add_vectors, 
        "Add two float32 NumPy vectors (CPU)"
    );
}