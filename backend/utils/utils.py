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

#ストレージサービスの代用（s3のようなもの)となるフォルダを作成する
#将来的にはストレージサービスに画像をアップロードする
def create_origin_image_folder(parent_path):
    uuid_string = uuid.uuid4()
    os.mkdir(f'{parent_path}/{uuid_string}')
    return uuid_string

def create_s3_prefix(prefix_str):
    # フォルダのように見えるプレフィックスを使用して空のオブジェクトを作成
    # S3クライアントの作成
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
    try:
    # 空のオブジェクトを作成
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=prefix_str)
        return True
    except Exception as e:
        print(e)
        return False