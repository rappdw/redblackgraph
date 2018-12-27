#!/usr/bin/env bash
set -e

if [ "$#" -eq 0 ]; then
    echo "You must specify a notebook from which to generate pdf"
    exit -1
fi

FILENAME=$(printf %q "$1")

build-image notebook
run-image -c "gen-latex.sh $FILENAME" notebook

dirname=$"build/latex-$(date '+%y-%m-%d.%s')"
mkdir -p $dirname

pushd notebooks
file_name="$(basename "$1" .ipynb)"
popd

echo docker run --rm --user $UID:$GID -v $PWD/notebooks:/sources -v $PWD:/output embix/pdflatex:v1 -output-directory="/output/$dirname" "$file_name.tex"
docker run --rm --user $UID:$GID -v $PWD/notebooks:/sources -v $PWD:/output embix/pdflatex:v1 -output-directory="/output/$dirname" "$file_name.tex"
