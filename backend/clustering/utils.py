from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import uuid
import base64

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
    