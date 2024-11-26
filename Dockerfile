FROM python:3-slim
ARG version

RUN apt-get update && apt-get install -y git gettext-base

ADD . /gracc-osg-reports
RUN git clone https://github.com/ashtongraves/gracc-reporting.git /gracc-reporting
WORKDIR /gracc-reporting
RUN pip install -r requirements.txt
RUN python setup.py install
WORKDIR /gracc-osg-reports
RUN pip install -r requirements.txt
RUN python setup.py install

RUN mkdir /tmp/html_templates && mkdir /tmp/gracc-osg-reports-config

RUN cp /gracc-osg-reports/html_templates/* /tmp/html_templates