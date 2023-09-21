ARG BUILD_FROM
FROM ${BUILD_FROM}

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        wget \
        gcc \
        g++ \
        gfortran \
        libgfortran-9-dev \
        liblapack-dev libamd2 libcholmod3 libmetis-dev libsuitesparse-dev libnauty2-dev git \
        make pkgconf nano file mariadb-client


RUN pip install entsoe_py --break-system-packages
RUN pip install fastapi --break-system-packages
RUN pip install influxdb --break-system-packages
RUN pip install influxdb-client --break-system-packages
RUN pip install lightgbm --break-system-packages
RUN pip install matplotlib --break-system-packages
RUN pip install mip --break-system-packages
RUN pip install numpy --break-system-packages
RUN pip install pandas --break-system-packages
RUN pip install pytz --break-system-packages
RUN pip install PyYAML --break-system-packages
RUN pip install requests --break-system-packages
RUN pip install scikit_learn --break-system-packages
RUN pip install seaborn --break-system-packages
RUN pip install skforecast --break-system-packages
RUN pip install statsmodels --break-system-packages
RUN pip install uvicorn --break-system-packages

WORKDIR /root/build
RUN wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
RUN chmod a+x ./coinbrew
RUN ./coinbrew fetch Cbc@master --no-prompt
RUN mkdir -p /root/build/miplib
RUN ./coinbrew build Cbc@master --no-prompt --prefix=/root/build/miplib/ --tests=none --enable-cbc-parallel --enable-relocatable
RUN cd /root/build/miplib/lib && ln -s libCbc.so.0.0.0 libCbcSolver.so
RUN export PMIP_CBC_LIBRARY="/root/build/miplib/lib/libCbcSolver.so"
RUN export LD_LIBRARY_PATH="/root/build/miplib/lib/"
RUN echo 'export PMIP_CBC_LIBRARY="/root/build/miplib/lib/libCbcSolver.so"' >> ~/.bashrc
RUN echo 'LD_LIBRARY_PATH="/root/build/miplib/lib/"' >> ~/.bashrc

WORKDIR /
# Install requirements for add-on
COPY functions.py /
COPY main.py /
COPY run.sh /
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]


