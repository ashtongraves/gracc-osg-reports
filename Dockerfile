FROM python:3-slim

RUN apt-get update && apt-get install -y gettext-base

ADD html_templates /tmp/html_templates/
ADD requirements.txt /app/
ADD reports /app/reports/
WORKDIR /app
RUN pip install -r requirements.txt
WORKDIR /app/reports