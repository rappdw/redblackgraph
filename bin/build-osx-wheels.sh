#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

export CFLAGS=-Qunused-arguments
export CPPFLAGS=-Qunused-arguments

conda activate rgb-build-36
python -m pip install --upgrade pip
pip install -U numpy cython scipy
python setup.py bdist_wheel -d ./wheel
conda deactivate

conda activate rgb-build-37
python -m pip install --upgrade pip
pip install -U numpy cython scipy
python setup.py bdist_wheel -d ./wheel
conda deactivate

conda activate rgb-build-38
python -m pip install --upgrade pip
pip install -U numpy cython scipy
python setup.py bdist_wheel -d ./wheel
conda deactivate

conda activate rgb-build-39
python -m pip install --upgrade pip
pip install -U numpy cython scipy
python setup.py bdist_wheel -d ./wheel
conda deactivate

conda activate rgb-build-310
python -m pip install --upgrade pip
pip install -U numpy cython scipy
python setup.py bdist_wheel -d ./wheel
conda deactivate
