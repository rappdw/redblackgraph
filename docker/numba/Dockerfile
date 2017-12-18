FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

ADD docker/numba/llvm.sources.list /etc/apt/llvm.sources.list

RUN cat /etc/apt/llvm.sources.list >>/etc/apt/sources.list

RUN apt-get update && apt-get install -y --allow-unauthenticated \
        build-essential \
        checkinstall \
        git \
        libreadline-gplv2-dev  \
        libncursesw5-dev \
        libssl-dev \
        libsqlite3-dev \
        tk-dev \
        libgdbm-dev \
        libc6-dev \
        libbz2-dev \
        clang-5.0 \
        lldb-5.0 \
        lld-5.0 \
        less \
        locales \
        vim \
        wget \
    && apt-get clean \
    && rm -rf /var/tmp /tmp /var/lib/apt/lists/* \
    && mkdir -p /var/tmp /tmp \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

RUN wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz \
    && tar xzf Python-3.6.3.tgz \
    && cd Python-3.6.3 \
    && ./configure --enable-optimizations \
    && make install

# make some useful symlinks that are expected to exist
#    && rm idle pydoc python python-config \
RUN ldconfig \
    && cd /usr//bin \
    && rm pydoc python \
	&& ln -Fs /usr/local/bin/idle3 idle \
	&& ln -Fs /usr/local/bin/pydoc3 pydoc \
	&& ln -Fs /usr/local/bin/python3 python \
	&& ln -Fs /usr/local/bin/python3-config python-config \
	&& ln -Fs /usr/local/bin/pip3 pip

ENV LLVM_CONFIG /usr/bin/llvm-config-5.0

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ADD requirements-numba.txt /requirements-numba.txt
RUN pip install -r /requirements-numba.txt

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /workdir
