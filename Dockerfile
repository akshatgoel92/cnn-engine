FROM python:3

MAINTAINER Akshat Goel

WORKDIR /app

COPY './requirements.txt' .

# RUN apt-get install libgtk2.0-dev pkg-config -yqq 

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY './models' .

COPY './templates' .

COPY './static' .

COPY './heroku.yml' .

COPY './app.py' .

COPY './assets' .

COPY './Procfile' .

COPY './train' .

CMD ["python", "app.py"]
