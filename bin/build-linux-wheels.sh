#!/usr/bin/env bash

mkdir /tmp/wheel
/opt/python/cp36-cp36m/bin/python -m pip install --upgrade pip
/opt/python/cp36-cp36m/bin/pip install numpy cython scipy
/opt/python/cp36-cp36m/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*

/opt/python/cp37-cp37m/bin/python -m pip install --upgrade pip
/opt/python/cp37-cp37m/bin/pip install numpy cython scipy
/opt/python/cp37-cp37m/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*

/opt/python/cp38-cp38/bin/python -m pip install --upgrade pip
/opt/python/cp38-cp38/bin/pip install numpy cython scipy
/opt/python/cp38-cp38/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*

/opt/python/cp39-cp39/bin/python -m pip install --upgrade pip
/opt/python/cp39-cp39/bin/pip install numpy cython scipy
/opt/python/cp39-cp39/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*

/opt/python/cp310-cp310/bin/python -m pip install --upgrade pip
/opt/python/cp310-cp310/bin/pip install numpy cython scipy
/opt/python/cp310-cp310/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*
