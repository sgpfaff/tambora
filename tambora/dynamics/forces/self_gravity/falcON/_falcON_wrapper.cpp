/// Python wrapper for falcON gravity solver from NEMO (W. Dehnen, ApJL 536, 9, 2000)
/// Provides: gravity(pos, mass, eps, theta=0.6, kernel=1) -> (acc, pot)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdexcept>
#include "forces.h"

// ---------------------------------------------------------------------------
// Helper: access element of a 1D numpy array
// ---------------------------------------------------------------------------
template<typename T>
inline T& elem1(void* arr, npy_intp i)
{
    return *static_cast<T*>(PyArray_GETPTR1(static_cast<PyArrayObject*>(arr), i));
}

// Helper: access element of a 2D numpy array
template<typename T>
inline T& elem2(void* arr, npy_intp i, npy_intp j)
{
    return *static_cast<T*>(PyArray_GETPTR2(static_cast<PyArrayObject*>(arr), i, j));
}

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
    "theta : float, optional\n"
    "    Tree opening angle (default 0.6). Smaller = more accurate.\n"
    "kernel : int, optional\n"
    "    Softening kernel: 0=Plummer, 1=default (~r^-7), 2,3=faster decay.\n\n"
    "Returns\n"
    "-------\n"
    "acc : (N, 3) float64 array\n"
    "    Gravitational accelerations.\n"
    "pot : (N,) float64 array\n"
    "    Gravitational potential at each particle.\n";

static PyObject* gravity(PyObject* /*self*/, PyObject* args, PyObject* kwds) {
    // TODO: parse arguments, build bodies, run falcON, return results
    // We will fill this in step by step.
    static const char* kwlist[] = {"pos", "mass", "eps", "theta", "kernel", NULL};
    PyObject *pos_obj = NULL, *mass_obj = NULL, *eps_obj = NULL;
    double theta = 0.6;
    int kernel = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|di", const_cast<char**>(kwlist),
                                     &pos_obj, &mass_obj, &eps_obj, &theta, &kernel)) {
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

    unsigned nbodies_arr[falcON::bodytype::NUM] = {0};
    nbodies_arr[falcON::bodytype::std] = unsigned(nbody);
    falcON::bodies B(nbodies_arr, falcON::fieldset(
        falcON::fieldset::gravity |                     // pos, mass, acc, pot, flags
        (eps_dtype != -1 ? falcON::fieldset::e : 0))    // individual eps if array
    );

    // load data into bodies
    falcON::body b = B.begin_all_bodies();
    for (npy_intp i = 0; i < nbody; ++i, ++b) {
        // position
        if (pos_dtype == NPY_FLOAT) {
            b.pos()[0] = elem2<float>(pos_obj, i, 0);
            b.pos()[1] = elem2<float>(pos_obj, i, 1);
            b.pos()[2] = elem2<float>(pos_obj, i, 2);
        }
        else {
            b.pos()[0] = elem2<double>(pos_obj, i, 0);
            b.pos()[1] = elem2<double>(pos_obj, i, 1);
            b.pos()[2] = elem2<double>(pos_obj, i, 2);
        }

        // mass
        if (mass_dtype == -1) //scalar
            b.mass() = mass_scalar;
        else if (mass_dtype == NPY_FLOAT)
            b.mass() = elem1<float>(mass_obj, i);
        else
            b.mass() = elem1<double>(mass_obj, i);

        // eps (only set if individual eps array was provided)
        if (eps_dtype == NPY_FLOAT)
            b.eps() = elem1<float>(eps_obj, i);
        else if (eps_dtype == NPY_DOUBLE)
            b.eps() = elem1<double>(eps_obj, i);

        b.flag_as_active();
    }

    PyObject *acc_arr = NULL, *pot_arr = NULL;

    try {
        falcON::forces F(&B,
            eps_scalar,                              // global softening length
            theta,                                   // tree opening angle
            (falcON::kern_type)kernel,               // softening kernel type
            eps_dtype != -1);                        // use individual eps_i?
        F.grow();
        F.approximate_gravity(true);
    } catch (std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }

    npy_intp dims_acc[2] = {nbody, 3};
    acc_arr = PyArray_SimpleNew(2, dims_acc, NPY_DOUBLE);

    npy_intp dims_pot[1] = {nbody};
    pot_arr = PyArray_SimpleNew(1, dims_pot, NPY_DOUBLE);

    b = B.begin_all_bodies();
    for (npy_intp i = 0; i < nbody; ++i, ++b) {
        elem2<double>(acc_arr, i, 0) = b.acc()[0];
        elem2<double>(acc_arr, i, 1) = b.acc()[1];
        elem2<double>(acc_arr, i, 2) = b.acc()[2];
        elem1<double>(pot_arr, i)    = b.pot();
    }
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
    "_falcon",                                   // module name (import as _falcon)
    "Python interface to falcON gravity solver",  // module docstring
    -1,                                           // module state size (-1 = global)
    methods
};

PyMODINIT_FUNC PyInit__falcon(void)
{
    PyObject* mod = PyModule_Create(&moduledef);
    import_array();  // initialize numpy C API — must be called once
    return mod;
}
