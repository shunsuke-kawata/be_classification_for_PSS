from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import uuid
import base64
import json
from io import BytesIO
from PIL import Image
import zipfile
import tempfile

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
    
    @classmethod
    def create_classification_folder_structure(cls, result_dict: dict, all_nodes_dict: dict, 
                                               source_images_path: Path, temp_dir: Path) -> Path:
        """
        分類結果に基づいてフォルダ構造を作成し、画像をコピーする
        
        Args:
            result_dict: MongoDBのresult（階層構造）
            all_nodes_dict: MongoDBのall_nodes（ノード情報）
            source_images_path: 元画像のパス
            temp_dir: 一時ディレクトリのパス
            
        Returns:
            Path: 作成されたフォルダ構造のルートパス
        """
        def create_folder_recursive(node_dict: dict, current_path: Path):
            """再帰的にフォルダ構造を作成"""
            for folder_id, folder_info in node_dict.items():
                folder_name = folder_info.get('name', folder_id)
                # ファイル名として使えない文字を置換
                safe_folder_name = "".join(c if c.isalnum() or c in (' ', '-', '_', '.') else '_' for c in folder_name)
                folder_path = current_path / safe_folder_name
                
                if folder_info.get('is_leaf', False):
                    # リーフフォルダ：画像をコピー
                    folder_path.mkdir(parents=True, exist_ok=True)
                    images_data = folder_info.get('data', {})
                    
                    for clustering_id, image_name in images_data.items():
                        source_image = source_images_path / image_name
                        if source_image.exists():
                            dest_image = folder_path / image_name
                            shutil.copy2(source_image, dest_image)
                        else:
                            print(f"⚠️ 画像が見つかりません: {source_image}")
                else:
                    # 非リーフフォルダ：再帰的に処理
                    folder_path.mkdir(parents=True, exist_ok=True)
                    sub_folders = folder_info.get('data', {})
                    if isinstance(sub_folders, dict):
                        create_folder_recursive(sub_folders, folder_path)
        
        # ルートフォルダから開始
        create_folder_recursive(result_dict, temp_dir)
        return temp_dir
    
    @classmethod
    def create_zip_from_folder(cls, folder_path: Path, zip_path: Path) -> Path:
        """
        フォルダをZIPファイルに圧縮する
        
        Args:
            folder_path: 圧縮するフォルダのパス
            zip_path: 作成するZIPファイルのパス
            
        Returns:
            Path: 作成されたZIPファイルのパス
        """
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = Path(root) / file
                    # ZIP内のパスを相対パスに
                    arcname = file_path.relative_to(folder_path)
                    zipf.write(file_path, arcname)
        
        return zip_path
    
    @classmethod
    def create_classification_download_package(cls, result_dict: dict, all_nodes_dict: dict,
                                               source_images_path: Path, project_name: str) -> Path:
        """
        分類結果のダウンロードパッケージを作成する
        
        Args:
            result_dict: MongoDBのresult
            all_nodes_dict: MongoDBのall_nodes
            source_images_path: 元画像のパス
            project_name: プロジェクト名（ZIPファイル名に使用）
            
        Returns:
            Path: 作成されたZIPファイルのパス
        """
        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            package_dir = temp_dir / project_name
            package_dir.mkdir(parents=True, exist_ok=True)
            
            # JSONファイルを保存
            result_json_path = package_dir / "result.json"
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            all_nodes_json_path = package_dir / "all_nodes.json"
            with open(all_nodes_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_nodes_dict, f, ensure_ascii=False, indent=2)
            
            # 画像フォルダ構造を作成
            images_dir = package_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            cls.create_classification_folder_structure(result_dict, all_nodes_dict, source_images_path, images_dir)
            
            # ZIPファイルを作成
            # 一時的なZIPファイルを/tmpに作成
            zip_filename = f"{project_name}.zip"
            zip_path = Path(tempfile.gettempdir()) / zip_filename
            
            cls.create_zip_from_folder(package_dir, zip_path)
            
            return zip_path