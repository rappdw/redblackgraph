#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

conda create -y --name rgb-build-36 python=3.6
conda create -y --name rgb-build-37 python=3.7
conda create -y --name rgb-build-38 python=3.8
conda create -y --name rgb-build-39 python=3.9
conda create -y --name rgb-build-310 python=3.10

docker pull quay.io/pypa/manylinux2010_x86_64

