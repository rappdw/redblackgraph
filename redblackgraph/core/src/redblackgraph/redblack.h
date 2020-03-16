#ifndef _REDBLACK_H_
#define _REDBLACK_H_

extern PyArrayObject * PyArray_EinsteinSum2(char *subscripts, npy_intp nop, PyArrayObject **op_in, PyArray_Descr *dtype, NPY_ORDER order, NPY_CASTING casting, PyArrayObject *out);

extern PyUFuncGenericFunction warshall_functions[];
extern void *warshall_data[];
extern char warshall_signatures[];
extern char *warshall_signature;

extern PyUFuncGenericFunction vertex_relational_composition_functions[];
extern void *vertex_relational_composition_data[];
extern char vertex_relational_composition_signatures[];
extern char *vertex_relational_composition_signature;

extern PyUFuncGenericFunction edge_relational_composition_functions[];
extern void *edge_relational_composition_data[];
extern char edge_relational_composition_signatures[];
extern char *edge_relational_composition_signature;

#endif