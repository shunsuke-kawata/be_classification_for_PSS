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

ALTER_STORAGE_SERVICE=os.environ['ALTER_STORAGE_SERVICE']