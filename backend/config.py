import os
from dotenv import load_dotenv
from enum import IntEnum

# .envファイルの内容を読み込む
load_dotenv()

#ポート設定
FRONTEND_PORT = os.environ['FRONTEND_PORT']
BACKEND_PORT = os.environ['BACKEND_PORT']

# MySQL 設定
DATABASE_PORT = os.environ['DATABASE_PORT']
DATABASE_PORT_IN_CONTAINER = os.environ['DATABASE_PORT_IN_CONTAINER']
MYSQL_ROOT_PASSWORD = os.environ['MYSQL_ROOT_PASSWORD']
MYSQL_DATABASE = os.environ['MYSQL_DATABASE']
MYSQL_USER = os.environ['MYSQL_USER']
MYSQL_PASSWORD = os.environ['MYSQL_PASSWORD']
MYSQL_HOST = os.environ['MYSQL_HOST']

# MongoDB 設定
MONGO_USER = os.environ['MONGO_USER']
MONGO_PASSWORD = os.environ['MONGO_PASSWORD']
MONGO_HOST = os.environ['MONGO_HOST']
MONGO_PORT = os.environ['MONGO_PORT']
MONGO_DB = os.environ['MONGO_DB']
MONGO_AUTH_DB = os.environ['MONGO_AUTH_DB']

# 管理者コード
ADMINISTRATOR_CODE = os.environ['ADMINISTRATOR_CODE']

# OpenAI API キー
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# パス設定
DEFAULT_IMAGE_PATH = os.environ.get('DEFAULT_IMAGE_PATH', 'images')
DEFAULT_OUTPUT_PATH = os.environ.get('DEFAULT_OUTPUT_PATH', 'output')

class CLUSTERING_STATUS(IntEnum):
    NOT_EXECUTED = 0
    EXECUTING = 1
    FINISHED = 2
    FAILED = 3