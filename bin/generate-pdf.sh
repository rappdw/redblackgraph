#!/usr/bin/env bash

build-image notebook
run-image -c convert.sh notebook

dirname=$"build/latex-$(date '+%y-%m-%d.%s')"
mkdir -p $dirname

pdflatex -output-directory=$dirname notebooks/*.tex