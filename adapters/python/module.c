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
#include <gffx/capabilities.h>

#include <stdint.h>
#include <string.h>

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

static int gffx_dict_set_owned(PyObject *dictionary, const char *key, PyObject *value) {
    int result;
    if (value == NULL) return -1;
    result = PyDict_SetItemString(dictionary, key, value);
    Py_DECREF(value);
    return result;
}

static PyObject *gffx_py_runtime_capabilities(
    PyObject *self,
    PyObject *args,
    PyObject *keywords
) {
    static char *keyword_names[] = {"include_sensitive", NULL};
    int include_sensitive = 0;
    uint32_t probe_flags = GFFX_CAPABILITY_PROBE_FULL;
    gffx_capability_report report = {0};
    gffx_diagnostic_buffer diagnostic = GFFX_DIAGNOSTIC_INIT;
    gffx_capability_record *records = NULL;
    char *strings = NULL;
    char diagnostic_text[1024] = {0};
    PyObject *record_list = NULL;
    PyObject *result_dictionary = NULL;
    gffx_status status;
    uint64_t index;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywords,
            "|p:runtime_capabilities",
            keyword_names,
            &include_sensitive)) return NULL;
    if (include_sensitive) probe_flags |= GFFX_CAPABILITY_PROBE_INCLUDE_SENSITIVE;
    diagnostic.data = diagnostic_text;
    diagnostic.capacity_bytes = (uint64_t)sizeof(diagnostic_text);
    report.struct_size = (uint32_t)sizeof(report);
    report.abi_version = GFFX_ABI_VERSION;
    status = gffx_capabilities_probe(probe_flags, &report, &diagnostic);
    if (status != GFFX_STATUS_INSUFFICIENT_WORKSPACE && status != GFFX_STATUS_OK) {
        PyErr_Format(PyExc_RuntimeError, "GFFX capability probe failed (%u): %s",
                     (unsigned int)status, diagnostic_text);
        return NULL;
    }
    if (report.required_record_count > (uint64_t)(SIZE_MAX / sizeof(*records)) ||
        report.required_string_bytes > (uint64_t)SIZE_MAX) {
        PyErr_SetString(PyExc_OverflowError, "GFFX capability report is too large");
        return NULL;
    }
    records = (gffx_capability_record *)PyMem_Malloc(
        (size_t)(report.required_record_count * sizeof(*records))
    );
    strings = (char *)PyMem_Malloc((size_t)report.required_string_bytes);
    if ((records == NULL && report.required_record_count != UINT64_C(0)) ||
        (strings == NULL && report.required_string_bytes != UINT64_C(0))) {
        PyMem_Free(records);
        PyMem_Free(strings);
        return PyErr_NoMemory();
    }
    memset(records, 0, (size_t)(report.required_record_count * sizeof(*records)));
    memset(strings, 0, (size_t)report.required_string_bytes);
    report.records = records;
    report.record_capacity = report.required_record_count;
    report.strings = strings;
    report.string_capacity_bytes = report.required_string_bytes;
    status = gffx_capabilities_probe(probe_flags, &report, &diagnostic);
    if (status != GFFX_STATUS_OK) {
        PyMem_Free(records);
        PyMem_Free(strings);
        PyErr_Format(PyExc_RuntimeError, "GFFX capability probe failed (%u): %s",
                     (unsigned int)status, diagnostic_text);
        return NULL;
    }

    record_list = PyList_New((Py_ssize_t)report.record_count);
    if (record_list == NULL) goto error;
    for (index = UINT64_C(0); index < report.record_count; ++index) {
        const gffx_capability_record *record = &records[index];
        PyObject *record_dictionary = PyDict_New();
        PyObject *value = NULL;
        if (record_dictionary == NULL) goto error;
        if (gffx_dict_set_owned(record_dictionary, "category",
                                PyLong_FromUnsignedLong(record->category)) != 0 ||
            gffx_dict_set_owned(record_dictionary, "subject_index",
                                PyLong_FromUnsignedLong(record->subject_index)) != 0 ||
            gffx_dict_set_owned(record_dictionary, "key",
                                PyLong_FromUnsignedLong(record->key)) != 0 ||
            gffx_dict_set_owned(record_dictionary, "value_type",
                                PyLong_FromUnsignedLong(record->value_type)) != 0 ||
            gffx_dict_set_owned(record_dictionary, "flags",
                                PyLong_FromUnsignedLong(record->flags)) != 0 ||
            gffx_dict_set_owned(record_dictionary, "sensitive",
                                PyBool_FromLong(
                                    (record->flags & GFFX_CAPABILITY_RECORD_SENSITIVE) != 0u
                                )) != 0) {
            Py_DECREF(record_dictionary);
            goto error;
        }
        if (record->value_type == GFFX_CAPABILITY_VALUE_STRING) {
            if (record->string_size == UINT64_C(0) ||
                record->string_offset > report.string_size_bytes ||
                record->string_size > report.string_size_bytes - record->string_offset) {
                Py_DECREF(record_dictionary);
                PyErr_SetString(PyExc_RuntimeError, "GFFX returned an invalid string record");
                goto error;
            }
            if ((record->flags & GFFX_CAPABILITY_RECORD_SENSITIVE) != 0u &&
                !include_sensitive) {
                value = PyUnicode_FromString("<redacted>");
            } else {
                value = PyUnicode_FromStringAndSize(
                    strings + record->string_offset,
                    (Py_ssize_t)(record->string_size - UINT64_C(1))
                );
            }
        } else if (record->value_type == GFFX_CAPABILITY_VALUE_I64) {
            value = PyLong_FromLongLong(record->value_i64);
        } else if (record->value_type == GFFX_CAPABILITY_VALUE_BOOL) {
            value = PyBool_FromLong(record->value_u64 != UINT64_C(0));
        } else {
            value = PyLong_FromUnsignedLongLong(record->value_u64);
        }
        if (gffx_dict_set_owned(record_dictionary, "value", value) != 0) {
            Py_DECREF(record_dictionary);
            goto error;
        }
        (void)PyList_SetItem(record_list, (Py_ssize_t)index, record_dictionary);
    }

    result_dictionary = PyDict_New();
    if (result_dictionary == NULL) goto error;
    if (gffx_dict_set_owned(result_dictionary, "query_flags",
                            PyLong_FromUnsignedLong(report.query_flags)) != 0 ||
        gffx_dict_set_owned(result_dictionary, "result_flags",
                            PyLong_FromUnsignedLong(report.result_flags)) != 0 ||
        gffx_dict_set_owned(result_dictionary, "include_sensitive",
                            PyBool_FromLong(include_sensitive)) != 0 ||
        PyDict_SetItemString(result_dictionary, "records", record_list) != 0) goto error;
    Py_DECREF(record_list);
    PyMem_Free(records);
    PyMem_Free(strings);
    return result_dictionary;

error:
    Py_XDECREF(result_dictionary);
    Py_XDECREF(record_list);
    PyMem_Free(records);
    PyMem_Free(strings);
    return NULL;
}

PyDoc_STRVAR(
    gffx_runtime_capabilities_doc,
    "runtime_capabilities(include_sensitive=False)\n--\n\n"
    "Explicitly load optional providers/drivers and return typed capability records."
);

static PyMethodDef gffx_methods[] = {
    {"abi_version", gffx_py_abi_version, METH_NOARGS, gffx_abi_version_doc},
    {"limited_api_version", gffx_py_limited_api_version, METH_NOARGS, gffx_limited_api_doc},
    {"runtime_capabilities", (PyCFunction)gffx_py_runtime_capabilities,
     METH_VARARGS | METH_KEYWORDS, gffx_runtime_capabilities_doc},
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
