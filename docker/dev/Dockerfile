FROM python:3.6.3

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

###################
# End of Base Image - If you make changes above this line, please reflect them in all other Dockerfiles
###################

ADD requirements-dev.txt /requirements-dev.txt
RUN pip install -r /requirements-dev.txt

# add less and vim (for debugging in container)
RUN apt-get update && apt-get install -y \
        bzip2 \
        ca-certificates \
        less \
        locales \
        fonts-liberation \
        sudo \
        vim \
        wget \
    && apt-get clean && \
    rm -rf /var/tmp /var/lib/apt/lists/*

# don't add project to the container, rather mount the project directory into the container
# to allow for editing in the host environment and having those changes available live in
# container
VOLUME /workdir
WORKDIR /workdir

# setup jupyter
RUN jupyter contrib nbextension install --system \
    && jupyter nbextensions_configurator enable --system \
    && jupyter nbextension enable hide_input/main \
    && jupyter nbextension enable hide_input_all/main \
    && jupyter nbextension enable python-markdown/main

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure environment
ENV SHELL=/bin/bash \
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV HOME=/home/$NB_USER

ADD docker/dev/fix-permissions /usr/local/bin/fix-permissions
# Create jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    fix-permissions $HOME

USER $NB_USER
RUN jupyter nbextension enable hide_input/main \
    && jupyter nbextension enable hide_input_all/main \
    && jupyter nbextension enable python-markdown/main

# Setup work directory for backward-compatibility
RUN mkdir /home/$NB_USER/work && \
    fix-permissions /home/$NB_USER

USER root
EXPOSE 8888


# As the entry point, install an editable version into the python environment and then
# start the shell
ENTRYPOINT /usr/local/bin/pip install -e /workdir && /bin/bash

# Add local files as late as possible to avoid cache busting
COPY docker/dev/start.sh /usr/local/bin/
COPY docker/dev/start-notebook.sh /usr/local/bin/
COPY docker/dev/start-singleuser.sh /usr/local/bin/
COPY docker/dev/jupyter_notebook_config.py /etc/jupyter/
RUN fix-permissions /etc/jupyter/
