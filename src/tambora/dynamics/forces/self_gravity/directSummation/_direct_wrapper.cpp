// Direct Summation Wrapper
// O(N^2) pairwise gravity with Plummer softening

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------------
// The gravity function
// ---------------------------------------------------------------------------
static const char* gravity_doc =
    "Compute gravitational acceleration and potential for a set of particles.\n\n"
    "Parameters\n"
    "----------\n"
    "pos : (N, 3) array of float32 or float64\n"
    "    Particle positions.\n"
    "mass : (N,) array or scalar\n"
    "    Particle masses. Cannot be zero.\n"
    "eps : (N,) array or scalar\n"
    "    Softening length(s).\n"
    "Returns\n"
    "-------\n"
    "acc : (N, 3) float64 array\n"
    "    Gravitational accelerations.\n"
    "pot : (N,) float64 array\n"
    "    Gravitational potential at each particle.\n";

static PyObject* gravity(PyObject* /*self*/, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"pos", "mass", "eps", NULL};
    PyObject *pos_obj = NULL, *mass_obj = NULL, *eps_obj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO", const_cast<char**>(kwlist),
                                     &pos_obj, &mass_obj, &eps_obj)) {
        return NULL; // error already set by PyArg_ParseTupleAndKeywords
    }

    if (!PyArray_Check(pos_obj) || PyArray_NDIM((PyArrayObject*)pos_obj) != 2 || PyArray_DIM((PyArrayObject*)pos_obj, 1) != 3 
                                || (PyArray_TYPE((PyArrayObject*)pos_obj) != NPY_DOUBLE && PyArray_TYPE((PyArrayObject*)pos_obj) != NPY_FLOAT)) {
        PyErr_SetString(PyExc_TypeError, "pos must be a 2D array of shape (N, 3) with dtype float32 or float64");
        return NULL;
    }
    npy_intp nbody = PyArray_DIM((PyArrayObject*)pos_obj, 0); // number of particles
    int pos_dtype = PyArray_TYPE((PyArrayObject*)pos_obj); // data type of pos (NPY_DOUBLE or NPY_FLOAT)
    
    double mass_scalar = 0;
    int mass_dtype = -1;

    if (PyFloat_Check(mass_obj) || PyLong_Check(mass_obj)) {
        // mass is a scalar
        mass_scalar = PyFloat_AsDouble(mass_obj);
    }
    else if (PyArray_Check(mass_obj)) {
        // mass is an array
        if (PyArray_NDIM((PyArrayObject*)mass_obj) != 1 || PyArray_DIM((PyArrayObject*)mass_obj, 0) != nbody
            || (PyArray_TYPE((PyArrayObject*)mass_obj) != NPY_DOUBLE && PyArray_TYPE((PyArrayObject*)mass_obj) != NPY_FLOAT)) {
            PyErr_SetString(PyExc_TypeError, "mass array must have shape (N,)");
            return NULL;
        }
        mass_dtype = PyArray_TYPE((PyArrayObject*)mass_obj);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "mass must be a float or a 1D array");
        return NULL;
    }

    double eps_scalar = 0;
    int eps_dtype = -1;

    if (PyFloat_Check(eps_obj) || PyLong_Check(eps_obj)) {
        // eps is a scalar
        eps_scalar = PyFloat_AsDouble(eps_obj);
    }
    else if (PyArray_Check(eps_obj)) {
        // eps is an array
        if (PyArray_NDIM((PyArrayObject*)eps_obj) != 1 || PyArray_DIM((PyArrayObject*)eps_obj, 0) != nbody
            || (PyArray_TYPE((PyArrayObject*)eps_obj) != NPY_DOUBLE && PyArray_TYPE((PyArrayObject*)eps_obj) != NPY_FLOAT)) {
            PyErr_SetString(PyExc_TypeError, "eps array must have shape (N,)");
            return NULL;
        }
        eps_dtype = PyArray_TYPE((PyArrayObject*)eps_obj);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "eps must be a float or a 1D array");
        return NULL;
    }

    PyObject *acc_arr = NULL, *pot_arr = NULL;

    npy_intp dims_acc[2] = {nbody, 3};
    acc_arr = PyArray_ZEROS(2, dims_acc, NPY_DOUBLE,0);

    npy_intp dims_pot[1] = {nbody};
    pot_arr = PyArray_ZEROS(1, dims_pot, NPY_DOUBLE,0);

    double* acc = (double*)PyArray_DATA((PyArrayObject*)acc_arr);
    double* pot = (double*)PyArray_DATA((PyArrayObject*)pot_arr);

    // ---- build contiguous double arrays for pos, mass, eps ----
    // We must NOT Py_DECREF the borrowed refs (pos_obj, mass_obj, eps_obj)
    // from PyArg_ParseTupleAndKeywords.  Instead, track contiguous array
    // references separately and Py_XDECREF them at cleanup.
    PyArrayObject* pos_cont  = NULL;
    PyArrayObject* mass_cont = NULL;
    PyArrayObject* eps_cont  = NULL;

    double* pos_d = NULL;
    bool pos_allocated = false;
    if (pos_dtype == NPY_DOUBLE) {
        pos_cont = (PyArrayObject*)PyArray_ContiguousFromAny(pos_obj, NPY_DOUBLE, 2, 2);
        if (!pos_cont) { Py_DECREF(acc_arr); Py_DECREF(pot_arr); return NULL; }
        pos_d = (double*)PyArray_DATA(pos_cont);
    } else {
        // float32 -> copy to double
        pos_cont = (PyArrayObject*)PyArray_ContiguousFromAny(pos_obj, NPY_FLOAT, 2, 2);
        if (!pos_cont) { Py_DECREF(acc_arr); Py_DECREF(pot_arr); return NULL; }
        pos_d = (double*)malloc(3 * nbody * sizeof(double));
        if (!pos_d) { Py_DECREF(pos_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return PyErr_NoMemory(); }
        pos_allocated = true;
        float* fp = (float*)PyArray_DATA(pos_cont);
        for (npy_intp k = 0; k < 3*nbody; k++) pos_d[k] = (double)fp[k];
        Py_DECREF(pos_cont); pos_cont = NULL;  // done with float array
    }

    double* mass_d = NULL;
    bool mass_allocated = false;
    if (mass_dtype == -1) {
        // scalar mass — fill array
        mass_d = (double*)malloc(nbody * sizeof(double));
        if (!mass_d) { if (pos_allocated) free(pos_d); Py_XDECREF(pos_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return PyErr_NoMemory(); }
        mass_allocated = true;
        for (npy_intp k = 0; k < nbody; k++) mass_d[k] = mass_scalar;
    } else if (mass_dtype == NPY_DOUBLE) {
        mass_cont = (PyArrayObject*)PyArray_ContiguousFromAny(mass_obj, NPY_DOUBLE, 1, 1);
        if (!mass_cont) { if (pos_allocated) free(pos_d); Py_XDECREF(pos_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return NULL; }
        mass_d = (double*)PyArray_DATA(mass_cont);
    } else {
        // float32 -> copy to double
        mass_cont = (PyArrayObject*)PyArray_ContiguousFromAny(mass_obj, NPY_FLOAT, 1, 1);
        if (!mass_cont) { if (pos_allocated) free(pos_d); Py_XDECREF(pos_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return NULL; }
        mass_d = (double*)malloc(nbody * sizeof(double));
        if (!mass_d) { Py_DECREF(mass_cont); if (pos_allocated) free(pos_d); Py_XDECREF(pos_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return PyErr_NoMemory(); }
        mass_allocated = true;
        float* fp = (float*)PyArray_DATA(mass_cont);
        for (npy_intp k = 0; k < nbody; k++) mass_d[k] = (double)fp[k];
        Py_DECREF(mass_cont); mass_cont = NULL;
    }

    double* eps_d = NULL;
    bool eps_allocated = false;
    if (eps_dtype == -1) {
        // scalar eps — fill array
        eps_d = (double*)malloc(nbody * sizeof(double));
        if (!eps_d) { if (pos_allocated) free(pos_d); if (mass_allocated) free(mass_d); Py_XDECREF(pos_cont); Py_XDECREF(mass_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return PyErr_NoMemory(); }
        eps_allocated = true;
        for (npy_intp k = 0; k < nbody; k++) eps_d[k] = eps_scalar;
    } else if (eps_dtype == NPY_DOUBLE) {
        eps_cont = (PyArrayObject*)PyArray_ContiguousFromAny(eps_obj, NPY_DOUBLE, 1, 1);
        if (!eps_cont) { if (pos_allocated) free(pos_d); if (mass_allocated) free(mass_d); Py_XDECREF(pos_cont); Py_XDECREF(mass_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return NULL; }
        eps_d = (double*)PyArray_DATA(eps_cont);
    } else {
        eps_cont = (PyArrayObject*)PyArray_ContiguousFromAny(eps_obj, NPY_FLOAT, 1, 1);
        if (!eps_cont) { if (pos_allocated) free(pos_d); if (mass_allocated) free(mass_d); Py_XDECREF(pos_cont); Py_XDECREF(mass_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return NULL; }
        eps_d = (double*)malloc(nbody * sizeof(double));
        if (!eps_d) { Py_DECREF(eps_cont); if (pos_allocated) free(pos_d); if (mass_allocated) free(mass_d); Py_XDECREF(pos_cont); Py_XDECREF(mass_cont); Py_DECREF(acc_arr); Py_DECREF(pot_arr); return PyErr_NoMemory(); }
        eps_allocated = true;
        float* fp = (float*)PyArray_DATA(eps_cont);
        for (npy_intp k = 0; k < nbody; k++) eps_d[k] = (double)fp[k];
        Py_DECREF(eps_cont); eps_cont = NULL;
    }

    // ---- O(N^2) direct summation ----
    for (npy_intp i = 0; i < nbody; i++) {
        double ax = 0, ay = 0, az = 0, phi = 0;
        double xi = pos_d[3*i], yi = pos_d[3*i+1], zi = pos_d[3*i+2];
        double ei = eps_d[i];

        for (npy_intp j = 0; j < nbody; j++) {
            if (i == j) continue;
            double dx = pos_d[3*j]   - xi;
            double dy = pos_d[3*j+1] - yi;
            double dz = pos_d[3*j+2] - zi;
            double eij = 0.5 * (ei + eps_d[j]);
            double r2 = dx*dx + dy*dy + dz*dz + eij*eij;
            double rinv = 1.0 / sqrt(r2);
            double mrinv = mass_d[j] * rinv;
            double mrinv3 = mrinv * rinv * rinv;
            ax += mrinv3 * dx;
            ay += mrinv3 * dy;
            az += mrinv3 * dz;
            phi -= mrinv;
        }
        acc[3*i]   = ax;
        acc[3*i+1] = ay;
        acc[3*i+2] = az;
        pot[i]     = phi;
    }

    // cleanup
    if (pos_allocated)  free(pos_d);
    if (mass_allocated) free(mass_d);
    if (eps_allocated)  free(eps_d);
    Py_XDECREF(pos_cont);
    Py_XDECREF(mass_cont);
    Py_XDECREF(eps_cont);

    return Py_BuildValue("(NN)", acc_arr, pot_arr);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
static PyMethodDef methods[] = {
    {"gravity", (PyCFunction)gravity, METH_VARARGS | METH_KEYWORDS, gravity_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_direct_summation",                                   // module name (import as _direct_summation)
    "Python interface to direct summation gravity solver",  // module docstring
    -1,                                           // module state size (-1 = global)
    methods
};

PyMODINIT_FUNC PyInit__direct_summation(void) {
    PyObject* mod = PyModule_Create(&moduledef);
    import_array();  // initialize numpy C API — must be called once
    return mod;
}