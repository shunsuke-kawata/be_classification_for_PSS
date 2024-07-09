FROM python:3.12.4
USER root
WORKDIR /app

ENV TZ JST-9

COPY ./backend/requirements.txt .

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -r requirements.txt

#バックエンドの起動
ENTRYPOINT ["python","server.py"]