FROM python:3.6.5

ADD requirements.txt /requirements.txt
ADD requirements-dev.txt /requirements-dev.txt
RUN pip install -U pip \
    && pip install -r /requirements-dev.txt

# add the project to the container
WORKDIR /workdir
ADD . /workdir

# mount a volume for output
VOLUME /workdir/_testing_output
RUN pip install /workdir
