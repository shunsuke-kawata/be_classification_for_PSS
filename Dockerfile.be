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
    cmake \
    curl

COPY ./backend/requirements.txt .

RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt

# 継続的クラスタリングで使用するモデルを事前ダウンロード
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# spaCy英語モデルをダウンロード
RUN python -m spacy download en_core_web_md

# NLTK WordNetデータをダウンロード
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

ENTRYPOINT ["python", "server.py"]