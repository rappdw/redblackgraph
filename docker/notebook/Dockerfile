FROM rappdw/docker-ds

USER root
ADD requirements.txt /tmp/requirements.txt
ADD docker/notebook/gen-latex.* /usr/local/bin/
ADD docker/notebook/start-redblack-notebook.sh /usr/local/bin

RUN pip install -r /tmp/requirements.txt \
    && mkdir -p /home/jovyan/.jupyter/custom

ADD docker/notebook/custom.css /home/jovyan/.jupyter/custom

USER $NB_USER
CMD start-project-notebook.sh
