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

# install spike-supervision and run it as the container command
WORKDIR /workdir
ADD . /workdir
RUN pip install -e .
