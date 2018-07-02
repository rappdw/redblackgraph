FROM rappdw/docker-ds:cpu-only

ADD docker/notebook/gen-latex.* /usr/local/bin/

RUN chmod 1777 /tmp; \
    apt-get update; \
    apt-get install --no-install-recommends --allow-unauthenticated -y \
        pandoc \
    ; \
    apt-get clean; \
    rm -rf /var/tmp /tmp /var/lib/apt/lists/*; \
    mkdir /tmp

ADD docker/notebook/custom.css /home/jovyan/.jupyter/custom/custom.css

