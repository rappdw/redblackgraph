#!/usr/bin/env bash

mkdir /tmp/wheel
/opt/python/cp36-cp36m/bin/pip install numpy cython
/opt/python/cp36-cp36m/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*

/opt/python/cp37-cp37m/bin/pip install numpy cython
/opt/python/cp37-cp37m/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*

/opt/python/cp38-cp38/bin/pip install numpy cython
/opt/python/cp38-cp38/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*

/opt/python/cp39-cp39/bin/pip install numpy cython
/opt/python/cp39-cp39/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*

/opt/python/cp310-cp310/bin/pip install numpy cython
/opt/python/cp310-cp310/bin/pip wheel /workdir -w /tmp/wheel
auditwheel repair /tmp/wheel/RedBlackGraph*.whl -w /workdir/wheel
rm /tmp/wheel/*
