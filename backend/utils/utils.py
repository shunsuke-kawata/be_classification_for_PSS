import uuid
import os

#ストレージサービスの代用（s3のようなもの)となるフォルダを作成する
#将来的にはストレージサービスに画像をアップロードする
def create_origin_image_folder(parent_path):
    uuid_string = uuid.uuid4()
    os.mkdir(f'{parent_path}/{uuid_string}')
    return uuid_string
