import datetime
from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker

import sys
sys.path.append('../')
from config import MYSQL_ROOT_PASSWORD,MYSQL_DATABASE,MYSQL_HOST,DATABASE_PORT_IN_CONTAINER

CONNECT_STRING = f"mysql://root:{MYSQL_ROOT_PASSWORD}@{MYSQL_HOST}:{DATABASE_PORT_IN_CONTAINER}/{MYSQL_DATABASE}?charset=utf8mb4"

#データベースに接続するためのセッションを作成する
def create_connect_session():
    try: 
        engine = create_engine(CONNECT_STRING,pool_size=10,max_overflow=20, pool_timeout=30,pool_recycle=1800)
        Session = sessionmaker(bind=engine)
        session = Session()
        return session
    except Exception as e:
        return None

#SQL文字列からクエリを実行する
def execute_query(session, query_text):
    try:
        query = text(query_text)
        result = session.execute(query)
        session.commit()

        created_id = session.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        return result, created_id
    except Exception as e:
        print(e)
        session.rollback()
        return None, None
    finally:
        session.close()
        