FROM rappdw/docker-ds:latest

ADD requirements.txt /tmp/requirements.txt
ADD docker/notebook/gen-latex.* /usr/local/bin/
ADD docker/notebook/start-redblack-notebook.sh /usr/local/bin

RUN . /home/jovyan/.venvs/notebook/bin/activate \
    && pip install -r /tmp/requirements.txt \
    && mkdir -p /home/jovyan/.jupyter/custom

ADD docker/notebook/custom.css /home/jovyan/.jupyter/custom

CMD start-redblack-notebook.sh
