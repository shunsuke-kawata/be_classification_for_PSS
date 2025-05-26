import base64
from io import BytesIO
import uuid
import random, string
import boto3

import sys
import os
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import ACCESS_KEY_ID, SECRET_ACCESS_KEY, S3_BUCKET_NAME

def generate_uuid():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('utf-8')

#画像ファイルをpngに変換する
def image2png(byte_image):
        # バイトデータを PIL イメージに変換
    image = Image.open(BytesIO(byte_image))

    # イメージを PNG に変換して保持
    png_image_io = BytesIO()
    image.save(png_image_io, format='PNG')
    png_image_io.seek(0)

    return png_image_io.getvalue()