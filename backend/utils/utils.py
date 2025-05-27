import base64
from io import BytesIO
import uuid
import random, string

import sys
import os
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def generate_uuid():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('utf-8')

# 画像ファイルを最高画質でPNGに変換する
def image2png(byte_image):
    # バイトデータを PIL イメージに変換
    image = Image.open(BytesIO(byte_image))

    # PNGとして最高画質（圧縮率最小 = compress_level=0）で保存
    png_image_io = BytesIO()
    image.save(png_image_io, format='PNG', optimize=True, compress_level=0)
    png_image_io.seek(0)

    return png_image_io.getvalue()