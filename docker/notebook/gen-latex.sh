#!/usr/bin/env bash
set -e
cd /home/jovyan/project
ESCAPED_ARGS=$(printf "%q " "$@")
eval "gen-latex.py $ESCAPED_ARGS"