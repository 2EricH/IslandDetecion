FROM python:3.6-slim

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt
