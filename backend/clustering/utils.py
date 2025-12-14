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

EXPAMPLE_JSON_PATH = Path('captions_20251120170437.json') #すでに生成されているキャプションのデータ

# キャプションデータをモジュールレベルでキャッシュ（起動時に一度だけ読み込み）
_caption_cache = None
_caption_cache_lock = None

def _load_caption_cache():
    """キャプションデータを一度だけ読み込んでキャッシュ"""
    global _caption_cache, _caption_cache_lock
    if _caption_cache is None:
        from threading import Lock
        if _caption_cache_lock is None:
            _caption_cache_lock = Lock()
        with _caption_cache_lock:
            if _caption_cache is None:  # ダブルチェック
                try:
                    with open(EXPAMPLE_JSON_PATH, encoding='utf-8') as f:
                        _caption_cache = json.load(f)
                    print(f"✅ キャプションデータをキャッシュしました: {len(_caption_cache)} 件")
                except Exception as e:
                    print(f"⚠️ キャプションデータの読み込み失敗: {e}")
                    _caption_cache = {}
    return _caption_cache

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
        """
        ファイル名をキーとしてキャプション情報を取得（キャッシュ版）
        
        Args:
            path: 画像ファイル名（例: "scissors_74268663-58a1-4f18-a9c3-c406736266cc.png"）
            
        Returns:
            tuple: (is_success: bool, caption: str)
        """
        try:
            # キャッシュからデータを取得
            data = _load_caption_cache()
            
            # ファイル名をキーとして直接取得
            if path in data:
                item = data[path]
                return item.get("is_success", False), item.get("caption", "No caption found.")
            else:
                print(f"⚠️ キャプション取得失敗: ファイル '{path}' がキャプションデータに存在しません")
                return False, "No caption found."
                
        except Exception as e:
            print(f"❌ キャプション取得中に予期しないエラー: {e}")
            return False, f"Error: {str(e)}"
    
    @classmethod
    # 画像ファイルを適切な品質でPNGに変換する
    def image2png(cls, byte_image, max_dimension: int = 2048):
        """
        画像をPNG形式に変換（適切な圧縮とリサイズを適用）
        
        Args:
            byte_image: 画像のバイトデータ
            max_dimension: 最大画像サイズ（幅・高さの最大値、デフォルト2048px）
        
        Returns:
            PNG形式の画像バイトデータ
        """
        # バイトデータを PIL イメージに変換
        image = Image.open(BytesIO(byte_image))
        
        # 大きすぎる画像はリサイズ（アスペクト比維持）
        # embedding生成時は224x224にリサイズされるため、事前に縮小しても問題なし
        if image.size[0] > max_dimension or image.size[1] > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        # PNGとして適切な圧縮レベル（6）で保存
        # compress_level=6: バランスの良い圧縮（0=無圧縮/最大サイズ, 9=最大圧縮/最遅）
        png_image_io = BytesIO()
        image.save(png_image_io, format='PNG', compress_level=6)
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