FROM rappdw/docker-ds:latest

ADD requirements.txt /tmp/requirements.txt
ADD docker/notebook/gen-latex.* /usr/local/bin/

RUN /bin/bash -c "source /home/jovyan/.venvs/notebook/bin/activate; pip install -r /tmp/requirements.txt" \
    && ln -s /workdir/notebooks notebooks \
    && mkdir -p /home/jovyan/.jupyter/custom

ADD docker/notebook/custom.css /home/jovyan/.jupyter/custom

# install an editable version into the python module and then, start the jupyter server
CMD /bin/bash -c "source /home/jovyan/.venvs/notebook/bin/activate; pip install -e /workdir/; start-notebook.sh"
