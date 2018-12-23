FROM python:3.6.5

# add the project to the container
WORKDIR /workdir
ADD . /workdir

# mount a volume for output
VOLUME /workdir/_testing_output
RUN pip install -U pip; \
    pip install "/workdir[dev]"
