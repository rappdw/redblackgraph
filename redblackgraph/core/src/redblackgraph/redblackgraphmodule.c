/*
 * The python module definiton for the Numpy impl of RedBlackGraph
 *
 * Copyright (c) 2017 by Dan Rapp (rappdw@gmail.com)
 *
 * See LICENSE.txt for the license.
 */


#include <Python.h>
#include <math.h>

#define PY_ARRAY_UNIQUE_SYMBOL RBG_ARRAY_API

#include <numpy/npy_3kcompat.h>
#include <numpy/noprefix.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

#include "rbg_math.h"
#include "redblack.h"

// ************ Compare with source in numpy/core/src/multiarray/mutliarraymodule.c ************

static int
einsum_sub_op_from_str(PyObject *args, PyObject **str_obj, char **subscripts,
                       PyArrayObject **op)
{
    int i, nop;
    PyObject *subscripts_str;

    nop = PyTuple_GET_SIZE(args) - 1;
    if (nop <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError, "too many operands");
        return -1;
    }

    /* Get the subscripts string */
    subscripts_str = PyTuple_GET_ITEM(args, 0);
    if (PyUnicode_Check(subscripts_str)) {
        *str_obj = PyUnicode_AsASCIIString(subscripts_str);
        if (*str_obj == NULL) {
            return -1;
        }
        subscripts_str = *str_obj;
    }

    *subscripts = PyBytes_AsString(subscripts_str);
    if (*subscripts == NULL) {
        Py_XDECREF(*str_obj);
        *str_obj = NULL;
        return -1;
    }

    /* Set the operands to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
    }

    /* Get the operands */
    for (i = 0; i < nop; ++i) {
        PyObject *obj = PyTuple_GET_ITEM(args, i+1);

        op[i] = (PyArrayObject *)PyArray_FROM_OF(obj, NPY_ARRAY_ENSUREARRAY);
        if (op[i] == NULL) {
            goto fail;
        }
    }

    return nop;

fail:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }

    return -1;
}

/*
 * Converts a list of subscripts to a string.
 *
 * Returns -1 on error, the number of characters placed in subscripts
 * otherwise.
 */
static int
einsum_list_to_subscripts(PyObject *obj, char *subscripts, int subsize)
{
    int ellipsis = 0, subindex = 0;
    npy_intp i, size;
    PyObject *item;

    obj = PySequence_Fast(obj, "the subscripts for each operand must "
                               "be a list or a tuple");
    if (obj == NULL) {
        return -1;
    }
    size = PySequence_Size(obj);


    for (i = 0; i < size; ++i) {
        item = PySequence_Fast_GET_ITEM(obj, i);
        /* Ellipsis */
        if (item == Py_Ellipsis) {
            if (ellipsis) {
                PyErr_SetString(PyExc_ValueError,
                        "each subscripts list may have only one ellipsis");
                Py_DECREF(obj);
                return -1;
            }
            if (subindex + 3 >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                Py_DECREF(obj);
                return -1;
            }
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            ellipsis = 1;
        }
        /* Subscript */
        else if (PyInt_Check(item) || PyLong_Check(item)) {
            long s = PyInt_AsLong(item);
            npy_bool bad_input = 0;

            if (subindex + 1 >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                Py_DECREF(obj);
                return -1;
            }

            if ( s < 0 ) {
                bad_input = 1;
            }
            else if (s < 26) {
                subscripts[subindex++] = 'A' + (char)s;
            }
            else if (s < 2*26) {
                subscripts[subindex++] = 'a' + (char)s - 26;
            }
            else {
                bad_input = 1;
            }

            if (bad_input) {
                PyErr_SetString(PyExc_ValueError,
                        "subscript is not within the valid range [0, 52)");
                Py_DECREF(obj);
                return -1;
            }
        }
        /* Invalid */
        else {
            PyErr_SetString(PyExc_ValueError,
                    "each subscript must be either an integer "
                    "or an ellipsis");
            Py_DECREF(obj);
            return -1;
        }
    }

    Py_DECREF(obj);

    return subindex;
}

/*
 * Fills in the subscripts, with maximum size subsize, and op,
 * with the values in the tuple 'args'.
 *
 * Returns -1 on error, number of operands placed in op otherwise.
 */
static int
einsum_sub_op_from_lists(PyObject *args,
                char *subscripts, int subsize, PyArrayObject **op)
{
    int subindex = 0;
    npy_intp i, nop;

    nop = PyTuple_Size(args)/2;

    if (nop == 0) {
        PyErr_SetString(PyExc_ValueError, "must provide at least an "
                        "operand and a subscripts list to einsum");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError, "too many operands");
        return -1;
    }

    /* Set the operands to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
    }

    /* Get the operands and build the subscript string */
    for (i = 0; i < nop; ++i) {
        PyObject *obj = PyTuple_GET_ITEM(args, 2*i);
        int n;

        /* Comma between the subscripts for each operand */
        if (i != 0) {
            subscripts[subindex++] = ',';
            if (subindex >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                goto fail;
            }
        }

        op[i] = (PyArrayObject *)PyArray_FROM_OF(obj, NPY_ARRAY_ENSUREARRAY);
        if (op[i] == NULL) {
            goto fail;
        }

        obj = PyTuple_GET_ITEM(args, 2*i+1);
        n = einsum_list_to_subscripts(obj, subscripts+subindex,
                                      subsize-subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    /* Add the '->' to the string if provided */
    if (PyTuple_Size(args) == 2*nop+1) {
        PyObject *obj;
        int n;

        if (subindex + 2 >= subsize) {
            PyErr_SetString(PyExc_ValueError,
                    "subscripts list is too long");
            goto fail;
        }
        subscripts[subindex++] = '-';
        subscripts[subindex++] = '>';

        obj = PyTuple_GET_ITEM(args, 2*nop);
        n = einsum_list_to_subscripts(obj, subscripts+subindex,
                                      subsize-subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    /* NULL-terminate the subscripts string */
    subscripts[subindex] = '\0';

    return nop;

fail:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }

    return -1;
}

static PyObject *
array_einsum2(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    char *subscripts = NULL, subscripts_buffer[256];
    PyObject *str_obj = NULL, *str_key_obj = NULL;
    PyObject *arg0;
    int i, nop;
    PyArrayObject *op[NPY_MAXARGS];
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    PyArrayObject *out = NULL;
    PyArray_Descr *dtype = NULL;
    PyObject *ret = NULL;

    if (PyTuple_GET_SIZE(args) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand, or at least one operand "
                        "and its corresponding subscripts list");
        return NULL;
    }
    arg0 = PyTuple_GET_ITEM(args, 0);

    /* einsum('i,j', a, b), einsum('i,j->ij', a, b) */
    if (PyString_Check(arg0) || PyUnicode_Check(arg0)) {
        nop = einsum_sub_op_from_str(args, &str_obj, &subscripts, op);
    }
    /* einsum(a, [0], b, [1]), einsum(a, [0], b, [1], [0,1]) */
    else {
        nop = einsum_sub_op_from_lists(args, subscripts_buffer,
                                    sizeof(subscripts_buffer), op);
        subscripts = subscripts_buffer;
    }
    if (nop <= 0) {
        goto finish;
    }

    /* Get the keyword arguments */
    if (kwds != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwds, &pos, &key, &value)) {
            char *str = NULL;

            Py_XDECREF(str_key_obj);
            str_key_obj = PyUnicode_AsASCIIString(key);
            if (str_key_obj != NULL) {
                key = str_key_obj;
            }

            str = PyBytes_AsString(key);

            if (str == NULL) {
                PyErr_Clear();
                PyErr_SetString(PyExc_TypeError, "invalid keyword");
                goto finish;
            }

            if (strcmp(str,"out") == 0) {
                if (PyArray_Check(value)) {
                    out = (PyArrayObject *)value;
                }
                else {
                    PyErr_SetString(PyExc_TypeError,
                                "keyword parameter out must be an "
                                "array for einsum");
                    goto finish;
                }
            }
            else if (strcmp(str,"order") == 0) {
                if (!PyArray_OrderConverter(value, &order)) {
                    goto finish;
                }
            }
            else if (strcmp(str,"casting") == 0) {
                if (!PyArray_CastingConverter(value, &casting)) {
                    goto finish;
                }
            }
            else if (strcmp(str,"dtype") == 0) {
                if (!PyArray_DescrConverter2(value, &dtype)) {
                    goto finish;
                }
            }
            else {
                PyErr_Format(PyExc_TypeError,
                            "'%s' is an invalid keyword for einsum",
                            str);
                goto finish;
            }
        }
    }

    ret = (PyObject *)PyArray_EinsteinSum2(subscripts, nop, op, dtype,
                                        order, casting, out);

    if (PyErr_Occurred()) {
        ret = NULL;
        goto finish;
    }

    /* If no output was supplied, possibly convert to a scalar */
    if (ret != NULL && out == NULL) {
        ret = PyArray_Return((PyArrayObject *)ret);
    }

finish:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
    }
    Py_XDECREF(dtype);
    Py_XDECREF(str_obj);
    Py_XDECREF(str_key_obj);
    /* out is a borrowed reference */

    return ret;
}

static PyObject *
c_avos_sum_test(PyObject *self, PyObject *args) {
    // It appears that some compilers (g++) emit a spurious warning on comparison of unsigned promotion.
    // This test case ensures that we are correct for all types we operate on.

    npy_byte a = -1;
    npy_byte b = 1;
    npy_byte c = byte_avos_sum(a, b);
    if (c != -1) {
        PyErr_Format(PyExc_ValueError, "Byte avos sum returned incorrect results");
        return NULL;
    }

    npy_ubyte a1 = -1;
    npy_ubyte b1 = 1;
    npy_ubyte c1 = ubyte_avos_sum(a1, b1);
    if (c1 != (npy_ubyte)-1) {
        PyErr_Format(PyExc_ValueError, "UByte avos sum returned incorrect results");
        return NULL;
    }

    npy_short a2 = -1;
    npy_short b2 = 1;
    npy_short c2 = short_avos_sum(a2, b2);
    if (c2 != -1) {
        PyErr_Format(PyExc_ValueError, "Short avos sum returned incorrect results");
        return NULL;
    }

    npy_ushort a3 = -1;
    npy_ushort b3 = 1;
    npy_ushort c3 = ushort_avos_sum(a3, b3);
    if (c3 != (npy_ushort)-1) {
        PyErr_Format(PyExc_ValueError, "UShort avos sum returned incorrect results");
        return NULL;
    }

    npy_int a4 = -1;
    npy_int b4 = 1;
    npy_int c4 = int_avos_sum(a4, b4);
    if (c4 != -1) {
        PyErr_Format(PyExc_ValueError, "Int avos sum returned incorrect results");
        return NULL;
    }

    npy_uint a5 = -1;
    npy_uint b5 = 1;
    npy_uint c5 = uint_avos_sum(a5, b5);
    if (c5 != (npy_uint)-1) {
        PyErr_Format(PyExc_ValueError, "UInt avos sum returned incorrect results");
        return NULL;
    }

    npy_long a6 = -1;
    npy_long b6 = 1;
    npy_long c6 = long_avos_sum(a6, b6);
    if (c6 != -1) {
        PyErr_Format(PyExc_ValueError, "Long avos sum returned incorrect results");
        return NULL;
    }

    npy_ulong a7 = -1;
    npy_ulong b7 = 1;
    npy_ulong c7 = ulong_avos_sum(a7, b7);
    if (c7 != (npy_ulong)-1) {
        PyErr_Format(PyExc_ValueError, "ULong avos sum returned incorrect results");
        return NULL;
    }

    npy_longlong a8 = -1;
    npy_longlong b8 = 1;
    npy_longlong c8 = longlong_avos_sum(a8, b8);
    if (c8 != -1) {
        PyErr_Format(PyExc_ValueError, "LongLong avos sum returned incorrect results");
        return NULL;
    }

    npy_ulonglong a9 = -1;
    npy_ulonglong b9 = 1;
    npy_ulonglong c9 = ulonglong_avos_sum(a9, b9);
    if (c9 != (npy_ulonglong)-1) {
        PyErr_Format(PyExc_ValueError, "ULongLong avos sum returned incorrect results");
        return NULL;
    }

    return PyLong_FromUnsignedLongLong(1);
}

static PyObject *
c_avos_sum_impl(PyObject *self, PyObject *args) {
    // given two arguments of python int (PyLong), use long_avos_sum to provide a result
    PyObject *arg0, *arg1;

    if (PyTuple_GET_SIZE(args) != 2) {
        PyErr_SetString(PyExc_ValueError, "two operands are required");
        return NULL;
    }

    arg0 = PyTuple_GET_ITEM(args, 0);
    arg1 = PyTuple_GET_ITEM(args, 1);

    npy_longlong l0 = PyLong_AsLongLong(arg0);
    npy_longlong l1 = PyLong_AsLongLong(arg1);
    npy_ulonglong result = ulonglong_avos_sum(l0, l1);

    if (result == NPY_MAX_ULONGLONG) return PyLong_FromLong(-1);
    return PyLong_FromUnsignedLongLong(result);
}

static PyObject *
c_avos_product_impl(PyObject *self, PyObject *args) {
    // given two arguments of python int (PyLong), use long_avos_product to provide a result
    PyObject *arg0, *arg1;

    if (PyTuple_GET_SIZE(args) != 2) {
        PyErr_SetString(PyExc_ValueError, "two operands are required");
        return NULL;
    }

    arg0 = PyTuple_GET_ITEM(args, 0);
    arg1 = PyTuple_GET_ITEM(args, 1);

    npy_longlong l0 = PyLong_AsLongLong(arg0);
    npy_longlong l1 = PyLong_AsLongLong(arg1);
    npy_ulonglong result = ulonglong_avos_product(l0, l1);

    if (PyErr_Occurred()) {
        return NULL;
    }

    if (result == NPY_MAX_ULONGLONG) return PyLong_FromLong(-1);
    return PyLong_FromUnsignedLongLong(result);
}

static struct PyMethodDef redblackgraph_module_methods[] = {
    {"c_einsum_avos",   (PyCFunction)array_einsum2,         METH_VARARGS|METH_KEYWORDS, "einsum avos function"},
    {"c_avos_sum",      (PyCFunction)c_avos_sum_impl,       METH_VARARGS,               "avos sum"},
    {"c_avos_product",  (PyCFunction)c_avos_product_impl,   METH_VARARGS,               "avos product"},
    {"c_avos_sum_test", (PyCFunction)c_avos_sum_test,       METH_VARARGS,               "test avos sum"},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_redblackgraph",
        NULL,
        -1,
        redblackgraph_module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
#define RETVAL m
PyMODINIT_FUNC PyInit__redblackgraph(void) {
    PyObject *m;
    PyObject *d;
    PyObject *f;

    /* Create the module and add the functions */
    m = PyModule_Create(&moduledef);
    if (!m) {
        goto err;
    }

#if defined(MS_WIN64) && defined(__GNUC__)
  PyErr_WarnEx(PyExc_Warning,
        "As with Numpy, RedBlackGraph built with MINGW-W64 on Windows 64 bits is experimental, " \
        "and only available for \n" \
        "testing. You are advised not to use it for production. \n\n" \
        "CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO REDBLACKGRAPH DEVELOPERS",
        1);
#endif

    import_array();
    import_umath();

    d = PyModule_GetDict(m);

    f = PyUFunc_FromFuncAndDataAndSignature(
            warshall_functions,                         // functions (len n_types)
            warshall_data,                              // data
            warshall_signatures,                        // types
            10,                                         // ntypes
            1,                                          // nin
            2,                                          // nout
            PyUFunc_None,                               // identity
            "warshall",                                 // name
            "warshall-floyd avos \n"
            "     \"(m,m)->(m,m),()\" \n",              // doc
            0,                                          // unused
            warshall_signature                          // signature
        );
    PyDict_SetItemString(d, "warshall", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
            vertex_relational_composition_functions,
            vertex_relational_composition_data,
            vertex_relational_composition_signatures,
            10,
            4,
            1,
            PyUFunc_None,
            "vertex_relational_composition",
            "avos vertex_relational_composition \n"
            "     \"(n),(n,n),(n),()->(n,n)\" \n",
            0,
            vertex_relational_composition_signature
        );
    PyDict_SetItemString(d, "vertex_relational_composition", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
            edge_relational_composition_functions,
            edge_relational_composition_data,
            edge_relational_composition_signatures,
            10,
            4,
            1,
            PyUFunc_None,
            "edge_relational_composition",
            "avos edge_relational_composition \n"
            "     \"(n),(),(),()->(n,n)\" \n",
            0,
            edge_relational_composition_signature
        );
    PyDict_SetItemString(d, "edge_relational_composition", f);
    Py_DECREF(f);

    if (PyErr_Occurred()) {
        goto err;
    }

    return RETVAL;

 err:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load _redblackgraph module.");
    }
    return RETVAL;
}

