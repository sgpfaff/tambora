#!/usr/bin/env python
import numpy as np
from setuptools import setup, Extension

_falcon_src_dir = "tambora/dynamics/forces/self_gravity/falcON/_falcON_src"

_falcon = Extension(
    name="tambora.dynamics.forces.self_gravity.falcON._falcon",
    sources=[
        "tambora/dynamics/forces/self_gravity/falcON/_falcON_wrapper.cpp",
        f"{_falcon_src_dir}/src/basic.cc",
        f"{_falcon_src_dir}/src/body.cc",
        f"{_falcon_src_dir}/src/gravity.cc",
        f"{_falcon_src_dir}/src/kernel.cc",
        f"{_falcon_src_dir}/src/tree.cc",
        f"{_falcon_src_dir}/src/exception.cc",
        f"{_falcon_src_dir}/src/numerics.cc",
        f"{_falcon_src_dir}/src/io.cc",
    ],
    include_dirs=[
        f"{_falcon_src_dir}/inc",
        f"{_falcon_src_dir}/inc/public",
        f"{_falcon_src_dir}/inc/utils",
        np.get_include(),
    ],
    define_macros=[
        ("falcON_DOUBLE", None),
    ],
    extra_compile_args=["-std=c++17"],
    language="c++",
)

_direct_summation = Extension(
    name="tambora.dynamics.forces.self_gravity.directSummation._direct_summation",
    sources=["tambora/dynamics/forces/self_gravity/directSummation/_direct_wrapper.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-std=c++17"],
    language="c++",
)

setup(ext_modules=[_falcon, _direct_summation])