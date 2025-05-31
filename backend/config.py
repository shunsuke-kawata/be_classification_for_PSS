import os
from dotenv import load_dotenv
from enum import IntEnum

# .envファイルの内容を読み込む
load_dotenv()

FRONTEND_PORT=os.environ['FRONTEND_PORT']
BACKEND_PORT =os.environ['BACKEND_PORT']
DATABASE_PORT=os.environ['DATABASE_PORT']
MYSQL_ROOT_PASSWORD=os.environ['MYSQL_ROOT_PASSWORD']
MYSQL_DATABASE=os.environ['MYSQL_DATABASE']
MYSQL_USER=os.environ['MYSQL_USER']
MYSQL_PASSWORD=os.environ['MYSQL_PASSWORD']
MYSQL_HOST=os.environ['MYSQL_HOST']
DATABASE_PORT_IN_CONTAINER=os.environ['DATABASE_PORT_IN_CONTAINER']
ADMINISTRATOR_CODE=os.environ['ADMINISTRATOR_CODE']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']

DEFAULT_IMAGE_PATH = os.environ.get('DEFAULT_IMAGE_PATH', 'images')
DEFAULT_OUTPUT_PATH = os.environ.get('DEFAULT_OUTPUT_PATH', 'output')


class CLUSTERING_STATUS(IntEnum):
    NOT_EXECUTED = 0
    EXECUTING = 1
    FINISHED = 2
    FAILED = 3