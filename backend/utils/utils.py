import uuid
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

#ストレージサービスの代用（s3のようなもの)となるフォルダを作成する
#将来的にはストレージサービスに画像をアップロードする
def create_origin_image_folder(parent_path):
    uuid_string = uuid.uuid4()
    os.mkdir(f'{parent_path}/{uuid_string}')
    return uuid_string
