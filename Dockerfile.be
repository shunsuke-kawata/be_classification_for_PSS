FROM python:3.12

WORKDIR /app

# Debian系のパッケージマネージャ apt を使用
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libmariadb-dev \
    libmariadb-dev-compat \
    default-libmysqlclient-dev \
    build-essential

COPY ./backend/requirements.txt .

RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "server.py"]