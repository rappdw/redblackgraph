#!/usr/bin/env bash

rm -rf ./wheel
mkdir -p ./wheel
mkdir -p ./wheel

bin/build-osx-wheels.sh

docker run --rm -it --mount type=bind,source=$(pwd),target=/workdir --entrypoint=/workdir/bin/build-linux-wheels.sh quay.io/pypa/manylinux_2_24_x86_64
docker run --rm -it --mount type=bind,source=$(pwd),target=/workdir --entrypoint=/workdir/bin/build-linux-wheels.sh quay.io/pypa/manylinux2014_x86_64