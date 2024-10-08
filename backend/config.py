import os
from dotenv import load_dotenv

FRONTEND_PORT=os.environ['FRONTEND_PORT']
BACKEND_PORT =os.environ['BACKEND_PORT']
DATABASE_PORT=os.environ['DATABASE_PORT']
MYSQL_ROOT_PASSWORD=os.environ['MYSQL_ROOT_PASSWORD']
MYSQL_DATABASE=os.environ['MYSQL_DATABASE']
MYSQL_USER=os.environ['MYSQL_USER']
MYSQL_PASSWORD=os.environ['MYSQL_PASSWORD']
ADMINISTRATOR_CODE=os.environ['ADMINISTRATOR_CODE']
ACCESS_KEY_ID=os.environ['ACCESS_KEY_ID']
SECRET_ACCESS_KEY=os.environ['SECRET_ACCESS_KEY']
S3_BUCKET_NAME=os.environ['S3_BUCKET_NAME']