FROM python:3.6.1

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

###################
# End of Base Image - If you make changes above this line, please reflect them in all other Dockerfiles
###################

# add less and vim (for debugging in container)
RUN apt-get update && apt-get install -y \
        less \
        vim \
    && apt-get clean && \
    rm -rf /var/tmp /var/lib/apt/lists/*

# don't add project to the container, rather mount the project directory into the container
# to allow for editing in the host environment and having those changes available live in
# container
VOLUME /workdir
WORKDIR /workdir

# As the entry point, install an editable version into the python environment and then
# start the shell
ENTRYPOINT /usr/local/bin/pip install -e /workdir && /bin/bash
