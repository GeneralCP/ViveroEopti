ARG BUILD_FROM
FROM ${BUILD_FROM}


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip

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


# Install requirements for add-on
COPY functions.py /
COPY main.py /
COPY run.sh /
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]


