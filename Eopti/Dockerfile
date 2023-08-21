FROM python:3.11.4-bookworm

WORKDIR /data
COPY requirements.txt /

RUN pip install entsoe_py
RUN pip install fastapi
RUN pip install influxdb
RUN pip install lightgbm
RUN pip install matplotlib
RUN pip install mip
RUN pip install numpy
RUN pip install pandas
RUN pip install pytz
RUN pip install PyYAML
RUN pip install requests
RUN pip install scikit_learn
RUN pip install seaborn
RUN pip install skforecast
RUN pip install statsmodels
RUN pip install uvicorn


# Install requirements for add-on
COPY functions.py /
COPY main.py /
COPY options.json /
COPY run.sh /
RUN chmod a+x /run.sh

CMD ["/run.sh"]


