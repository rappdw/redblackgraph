#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

conda activate rgb-build-36
pip install -U numpy
python setup.py bdist_wheel -d ./wheel
conda deactivate

conda activate rgb-build-37
pip install -U numpy
python setup.py bdist_wheel -d ./wheel
conda deactivate

conda activate rgb-build-38
pip install -U numpy
python setup.py bdist_wheel -d ./wheel
conda deactivate

