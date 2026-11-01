#!/usr/bin/env bash
# Generate a PDF of a notebook.
#
# Usage: bin/generate-pdf.sh "notebooks/Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.ipynb"
#
# Requires jupyter nbconvert (pip install nbconvert), pandoc, and a TeX installation that
# provides xelatex (e.g. TeX Live). The PDF is written to build/latex-<timestamp>/.
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "You must specify a notebook from which to generate pdf" >&2
    exit 1
fi

for tool in jupyter pandoc xelatex; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: '$tool' is required but was not found on PATH" >&2
        exit 1
    fi
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
notebook="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
notebook_dir="$(dirname "$notebook")"
name="$(basename "$notebook" .ipynb)"
out_dir="$repo_root/build/latex-$(date '+%y-%m-%d.%s')"
mkdir -p "$out_dir"

# convert to LaTeX and apply the fixups (writes "$name.tex" next to the notebook)
python "$repo_root/bin/gen-latex.py" "$notebook"

# compile from the notebook directory so relative image paths resolve; run twice for references
cd "$notebook_dir"
for _ in 1 2; do
    xelatex -interaction=nonstopmode -halt-on-error -output-directory="$out_dir" "$name.tex" >/dev/null
done
mv "$name.tex" "$out_dir/"

echo "Wrote $out_dir/$name.pdf"
