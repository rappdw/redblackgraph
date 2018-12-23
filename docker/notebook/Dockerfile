FROM rappdw/docker-ds

ADD docker/notebook/gen-latex.* /usr/local/bin/

RUN chmod 1777 /tmp; \
    apt-get update; \
    apt-get install --no-install-recommends --allow-unauthenticated -y \
        gcc \
        g++ \
        linux-headers-4.9-amd64 \
        libc6-dev \
        pandoc \
    ; \
    pip install -r /tmp/requirements.txt; \
    mkdir -p /home/jovyan/.jupyter/custom; \
    fix-permissions /home/jovyan/.jupyter; \
    apt-get clean; \
    rm -rf /var/tmp/* /tmp/* /var/lib/apt/lists/*

ADD docker/notebook/custom.css /home/jovyan/.jupyter/custom/custom.css

