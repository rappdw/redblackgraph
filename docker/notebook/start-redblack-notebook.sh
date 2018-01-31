#!/usr/bin/env bash

# get the password from credstash
. /home/jovyan/.venvs/notebook/bin/activate

# install an editable version of the ner module into site-packages
pip install -e /home/jovyan/redblackgraph/

# trust our notebooks
pushd /home/jovyan/redblackgraph/notebooks
for f in *.ipynb; do
    jupyter trust "$f"
done
popd

# start the notebook
start-notebook.sh