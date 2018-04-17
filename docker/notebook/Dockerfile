FROM rappdw/docker-ds

USER root
ADD requirements.txt /tmp/requirements.txt
ADD docker/notebook/gen-latex.* /usr/local/bin/

RUN chmod 1777 /tmp \
    && apt-get update && apt-get install --no-install-recommends --allow-unauthenticated -y \
    pandoc \
    && pip install -r /tmp/requirements.txt \
    && mkdir -p /home/jovyan/.jupyter/custom

ADD docker/notebook/custom.css /home/jovyan/.jupyter/custom

USER $NB_USER
