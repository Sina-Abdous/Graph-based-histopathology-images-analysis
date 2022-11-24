FROM sina-abdous-nvidia:latest
ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get update && apt-get install git -y sudo
#RUN apt-get install git -y sudo

#RUN apt-get install -y software-properties-common sudo
#RUN add-apt-repository -y ppa:deadsnakes/ppa sudo
#RUN apt-get update && apt-get install git sudo -y

#RUN apt-get install --no-install-recommends -y python3.9 python3.9-dev python3-pip python3-dev python-dev python-setuptools python3-setuptools
#RUN apt-get install -y python3.9-distutils
#RUN apt-get install -y gfortran libopenblas-dev liblapack-dev
#RUN apt-get install -y gcc g++

#RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

#sina-abdous-nvidiaRUN python -m pip install --upgrade wheel setuptools pip distlib

RUN mkdir -p /tokengt
RUN git clone --recursive https://github.com/Sina-Abdous/Graph-based-histopathology-images-analysis.git /tokengt
RUN git config --global --add safe.directory /tokengt
RUN git config --global --add safe.directory /tokengt/large-scale-regression/fairseq
WORKDIR /tokengt
RUN git checkout bracs

RUN bash install.sh
CMD ["/bin/bash"]
