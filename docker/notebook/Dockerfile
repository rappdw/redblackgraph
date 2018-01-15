FROM docker-ds-notebook:drapp

ADD requirements.txt /tmp/requirements.txt

RUN python -m venv --system-site-packages .venvs/notebook \
    && /bin/bash -c "source /home/jovyan/.venvs/notebook/bin/activate; pip install -r /tmp/requirements.txt" \
    && ln -s /workdir/notebooks notebooks

# install an editable version into the python module and then, start the jupyter server
CMD /bin/bash -c "source /home/jovyan/.venvs/notebook/bin/activate; pip install -e /workdir/; start-notebook.sh"
