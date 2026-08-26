/* Private PyTorch Stable-ABI loader and registration proof.
 *
 * Phase 1 intentionally registers no advertised graphics or geometry operation. The internal
 * schema proves that the CPython module's static registration runs when the adapter is loaded.
 */

#include <Python.h>
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(gffx_internal, m) {
    m.def("_foundation_probe() -> ()");
}

static struct PyModuleDef gffx_torch_module = {
    PyModuleDef_HEAD_INIT,
    "_torch",
    "Private GFFX PyTorch Stable-ABI registration loader.",
    0,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__torch(void) { return PyModule_Create(&gffx_torch_module); }
