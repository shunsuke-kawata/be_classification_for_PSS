FROM python:3.11

WORKDIR /app

# Debian系のパッケージマネージャ apt を使用
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libmariadb-dev \
    libmariadb-dev-compat \
    default-libmysqlclient-dev \
    build-essential \
    cmake

COPY ./backend/requirements.txt .

RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENTRYPOINT ["python", "server.py"]