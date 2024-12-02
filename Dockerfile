FROM python:3-slim
ARG version

RUN apt-get update && apt-get install -y gettext-base

ADD html_templates /tmp/html_templates
RUN mkdir /tmp/gracc-osg-reports-config
ADD setup.py requirements.txt /app/
ADD gracc_osg_reports /app/gracc_osg_reports/
WORKDIR /app
RUN pip install -r requirements.txt
RUN python setup.py install