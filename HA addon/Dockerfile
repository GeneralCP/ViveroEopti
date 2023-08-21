FROM python:3.11.4-bookworm

WORKDIR /app
COPY requirements.txt requirements.txt

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
COPY options.json options.json
COPY functions.py functions.py
COPY main.py main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


