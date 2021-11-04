FROM python:3.8-slim

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip && \
    apt-get clean

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

WORKDIR /app

ENTRYPOINT ["sh", "entrypoint.sh" ] 
