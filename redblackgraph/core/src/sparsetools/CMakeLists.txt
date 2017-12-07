cmake_minimum_required(VERSION 3.6)
project(sparsetools)

set(CMAKE_CXX_STANDARD 11)
set(NPY_NO_DEPRECATED_API 1)
set(NPY_1_9_API_VERSION 1)


set(SOURCE_FILES
        bool_ops.h
        complex_ops.h
        rbm.cxx
        rbm.h
        rbm_impl.h
        sparsetools.cxx
        sparsetools.h
        sparsetools_impl.h)

include_directories(/Users/drapp/sandbox/redBlackGraph/.venv/lib/python3.6/site-packages/numpy/core/include)
include_directories(/Users/drapp/sandbox/redBlackGraph/.venv/include)
include_directories(/Library/Frameworks/Python.framework/Versions/3.6/include/python3.6m)

add_executable(sparsetools ${SOURCE_FILES})