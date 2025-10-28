from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import uuid
import base64
import json
from io import BytesIO
from PIL import Image

EXPAMPLE_JSON_PATH = Path('captions_20250522_013210.json')

class Utils:
    
    @classmethod
    def copy_images_parallel(cls,metadata_list, src_folder, dest_folder):
        THREADS = 8
        def copy_one(metadata):
            src = src_folder / Path(metadata.path)
            if src.exists():
                dest = dest_folder / src.name
                shutil.copy(src, dest)

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            executor.map(copy_one, metadata_list)
            
    @classmethod
    def generate_uuid(cls):
        return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('utf-8')
        
    @classmethod
    def get_exmaple_caption(cls, path: str) -> dict:
        with open(EXPAMPLE_JSON_PATH, encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if item.get("path") == path:
                return item.get("is_success"),item.get("caption")
        
        return False,"No caption found."
    
    @classmethod
    # 画像ファイルを最高画質でPNGに変換する
    def image2png(cls,byte_image):
        # バイトデータを PIL イメージに変換
        image = Image.open(BytesIO(byte_image))

        # PNGとして最高画質（圧縮率最小 = compress_level=0）で保存
        png_image_io = BytesIO()
        image.save(png_image_io, format='PNG', optimize=True, compress_level=0)
        png_image_io.seek(0)

        return png_image_io.getvalue()
        
    