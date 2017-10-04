FROM python:3.6.3

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

###################
# End of Base Image - If you make changes above this line, please reflect them in all other Dockerfiles
###################

# add the project to the container
WORKDIR /workdir
ADD . /workdir

# mount a volume for output
VOLUME /workdir/_testing_output
