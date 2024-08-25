import uuid
import random, string
import boto3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import ACCESS_KEY_ID, SECRET_ACCESS_KEY, S3_BUCKET_NAME

#uuidのような文字列を生成する
def create_random_string(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)

#s3にプレフィックスを作成する
def create_s3_prefix(prefix_str:str,byte_data=None):
    # フォルダのように見えるプレフィックスを使用して空のオブジェクトを作成
    # S3クライアントの作成
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
    try:
    # 空のオブジェクトを作成
        if(byte_data is None):
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=prefix_str)
        else:
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=prefix_str, Body=byte_data)
        return True
    except Exception as e:
        print(e)
        return False

#画像ファイルをpngに変換する
def image2png(byte_image,origin_ext):
    return 