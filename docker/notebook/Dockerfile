FROM resero/docker-ds

ADD docker/notebook/gen-latex.* /usr/local/bin/

RUN set -e; \
    apt-get update; \
    apt-get install --no-install-recommends --allow-unauthenticated -y \
        gcc \
        g++ \
        linux-headers-amd64 \
        libc6-dev \
    ; \
    wget -q https://github.com/jgm/pandoc/releases/download/1.19.2.1/pandoc-1.19.2.1-1-amd64.deb; \
    dpkg -i pandoc-1.19.2.1-1-amd64.deb; \
    pip install "numpy>=2.0.0,<3.0" "scipy>=1.11.0" "XlsxWriter"; \
    mkdir -p /home/jovyan/.jupyter/custom; \
    fix-permissions /home/jovyan/.jupyter; \
    apt-get clean; \
    rm -rf /var/tmp/* /tmp/* /var/lib/apt/lists/*

ADD docker/notebook/custom.css /home/jovyan/.jupyter/custom/custom.css

