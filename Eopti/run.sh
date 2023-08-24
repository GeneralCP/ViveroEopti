#!/usr/bin/with-contenv bashio

CONFIG_PATH=/data/options.json

exec uvicorn main:app --host 0.0.0.0 --port 8000