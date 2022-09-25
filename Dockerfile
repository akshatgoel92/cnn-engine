FROM python:3

MAINTAINER Akshat Goel

WORKDIR /app

COPY './requirements.txt' .

#RUN apt-get install libgtk2.0-dev pkg-config -yqq 

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY './models' models

COPY './templates' templates

COPY './static' static

COPY './heroku.yml' .

COPY './app.py' .

COPY './assets' assets

COPY './Procfile' .

COPY './train' train

CMD ["python", "app.py"]
