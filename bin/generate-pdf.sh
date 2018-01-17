#!/usr/bin/env bash

build-image notebook
run-image -c "gen-latex.sh '$1'" notebook

dirname=$"build/latex-$(date '+%y-%m-%d.%s')"
mkdir -p $dirname

pushd notebooks
file_name="$(basename "$1" .ipynb)"
pdflatex -output-directory="../$dirname" "$file_name.tex"
popd