import json
from fastapi import APIRouter, status
import sys
import os
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from clustering.caption_manager import CaptionManager

#分割したエンドポイントの作成
test_endpoint = APIRouter()

#ユーザ一覧の取得
@test_endpoint.get('/test',tags=["users"],description="テストエンドポイントの追加")
def read_test():
    print("test endpoint called")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "test endpoint called successfully"})
