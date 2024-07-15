import json
from fastapi import APIRouter, status,Response
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.migrate import migration_engine,Base

#html出力のテンプレート
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <title>Classification for PSS</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
    </style>
    <style data-styled="" data-styled-version="4.4.1"></style>
</head>
<body>
    <div id="redoc-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"> </script>
    <script>
        var spec = %s;
        Redoc.init(spec, {}, document.getElementById("redoc-container"));
    </script>
</body>
</html>
"""

#分割したエンドポイントの作成
#デバックなどのための関数をおくエンドポイント
systems_endpoint = APIRouter()

@systems_endpoint.get('/system/db/migrate',tags=["systems"],description="データベースのマイグレーション処理を行う")
def migrate_db():
    try:
        Base.metadata.create_all(bind=migration_engine)
        return Response(status_code=status.HTTP_200_OK,content=json.dumps({"message":"succeeded to migrate database"}))
    except Exception as e:
        print(e)
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to migrate database"}))
