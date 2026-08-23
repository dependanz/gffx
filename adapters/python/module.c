/*
 * CPython Limited-API loading glue for the GFFX native core.
 *
 * Built against the CPython 3.10 Limited API (`Py_LIMITED_API`), so one compiled artifact is
 * valid on CPython 3.10 and every later version, and the wheel can carry the `abi3` tag. Only
 * stable-ABI symbols are used; the module must never import anything from a version-specific
 * CPython DLL.
 *
 * This module is private (`gffx._core`). It is loading glue, not a public API: it advertises no
 * graphics or geometry operation and exposes only the identity a capability report needs.
 */

/* The build system defines this via python_add_library(USE_SABI 3.10); the guard keeps a direct
 * compile of this file honest without redefining the macro. */
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030A0000
#endif

#include <Python.h>

#include <gffx/abi.h>

PyDoc_STRVAR(
    gffx_module_doc,
    "Private CPython Limited-API loader for the GFFX native core.\n"
    "Not a public API surface: no graphics or geometry operation is exposed here."
);

PyDoc_STRVAR(
    gffx_abi_version_doc,
    "abi_version()\n--\n\n"
    "Return the encoded native ABI version reported by the linked gffx_core library."
);

static PyObject *gffx_py_abi_version(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromUnsignedLong((unsigned long)gffx_get_abi_version());
}

PyDoc_STRVAR(
    gffx_limited_api_doc,
    "limited_api_version()\n--\n\n"
    "Return the Py_LIMITED_API floor this module was compiled against."
);

static PyObject *gffx_py_limited_api_version(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromUnsignedLong((unsigned long)Py_LIMITED_API);
}

static PyMethodDef gffx_methods[] = {
    {"abi_version", gffx_py_abi_version, METH_NOARGS, gffx_abi_version_doc},
    {"limited_api_version", gffx_py_limited_api_version, METH_NOARGS, gffx_limited_api_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gffx_module = {
    PyModuleDef_HEAD_INIT,
    "gffx._core",
    gffx_module_doc,
    -1,
    gffx_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&gffx_module);
}
