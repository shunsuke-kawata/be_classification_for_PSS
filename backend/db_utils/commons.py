import datetime
from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker

import sys
sys.path.append('../')
from config import MYSQL_ROOT_PASSWORD,DATABASE_PORT,MYSQL_DATABASE

CONNECT_STRING = f"mysql://root:{MYSQL_ROOT_PASSWORD}@db-pss-app:{DATABASE_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"

#データベースに接続するためのセッションを作成する
def create_connect_session():
    try: 
        engine = create_engine(CONNECT_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()
        return session
    except Exception as e:
        print(e)
        return None

#SQL文字列からクエリを実行する
def execute_query(session,query_text):
    try:
        query = text(query_text)
        result = session.execute(query)
        session.commit()
        
        # リソース作成時に使用する
        created_id = session.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        return result,created_id
    except Exception as e:
        print(e)
        session.rollback()
        return None,None