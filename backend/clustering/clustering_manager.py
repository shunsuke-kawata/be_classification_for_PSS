import json
import os
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from .chroma_db_manager import ChromaDBManager
import re
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from .embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from .utils import Utils
class InitClusteringManager:
    
    COHESION_THRESHOLD = 0.75
    
    def __init__(self, chroma_db: ChromaDBManager, images_folder_path: str, output_base_path: str = './results'):
        def _is_valid_path(path: str) -> bool:
            if not isinstance(path, str) or not path.strip():
                return False

            # 条件 3: 必ず ./ ../ / のいずれかで始まる
            if not (path.startswith("./") or path.startswith("../") or path.startswith("/")):
                return False

            # 条件 4: 最後は / で終わってはいけない
            if path.endswith("/"):
                return False

            # 条件 5: 危険文字の除外
            if re.search(r'[<>:"|?*]', path):
                return False

            return True

        if not (_is_valid_path(images_folder_path) and _is_valid_path(output_base_path)):
            raise ValueError(f" Error Folder Path: {images_folder_path}, {output_base_path}")
        
        self._chroma_db = chroma_db
        self._images_folder_path = Path(images_folder_path)
        self._output_base_path = Path(output_base_path)
    
    @property
    def chroma_db(self) -> ChromaDBManager:
        return self._chroma_db

    @property
    def images_folder_path(self) -> Path:
        return self._images_folder_path
    
    @property
    def output_base_path(self) -> Path:
        return self._output_base_path
    
    def get_optimal_cluster_num(self, embeddings: list[float], min_cluster_num: int = 5, max_cluster_num: int = 30) -> tuple[int, float]:
        embeddings_np = np.array(embeddings)
        n_samples = len(embeddings_np)

        if n_samples < 3:
            print("サンプル数が少なすぎてクラスタリングできません")
            return 1, -1.0

        best_score = -1
        best_k = min_cluster_num
        scores = []

        for k in range(min_cluster_num, max_cluster_num + 1):
            if k >= n_samples:
                continue  # クラスタ数がサンプル数以上は無効

            try:
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(embeddings_np)

                if len(set(labels)) < 2:
                    continue  # すべて同じクラスタ

                score = silhouette_score(embeddings_np, labels)
                scores.append((k, score))

                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"k={k} のときにエラーが発生: {e}")
                continue
        
        print("スコア一覧:", scores)

        return best_k, float(best_score) if best_score >= 0 else (1, -1.0)
    
    def clustering(self, chroma_db_data: dict[str, list], cluster_num: int, output_folder: bool = False, output_json: bool = False):
        embeddings_np = np.array(chroma_db_data['embeddings'])
        result_uuids_dict = {}

        if cluster_num <= 1:
            # すべてを1クラスタとして処理
            folder_id = Utils.generate_uuid()
            result_uuids_dict[0] = {'folder_id': folder_id, 'data': {}}

            for i in range(len(chroma_db_data['ids'])):
                result_uuids_dict[0]['data'][chroma_db_data['ids'][i]] = chroma_db_data['metadatas'][i].path

            if output_folder:
                output_dir = self._output_base_path / folder_id
                output_dir.mkdir(parents=True, exist_ok=True)
                Utils.copy_images_parallel(
                    chroma_db_data['metadatas'],
                    self._images_folder_path,
                    output_dir
                )
        
        else:
            # 通常通りクラスタリング
            model = AgglomerativeClustering(n_clusters=cluster_num)
            labels = model.fit_predict(embeddings_np)

            if output_folder:
                if self._output_base_path.exists():
                    shutil.rmtree(self._output_base_path)
                self._output_base_path.mkdir(parents=True, exist_ok=True)

            for idx in range(cluster_num):
                folder_id = Utils.generate_uuid()
                result_uuids_dict[idx] = {'folder_id': folder_id, 'data': {}}

            for i, label in enumerate(labels):
                result_uuids_dict[label]['data'][chroma_db_data['ids'][i]] = chroma_db_data['metadatas'][i].path

            # （中略）← 従来のサブクラスタリング処理（cohesionチェックなど）

        # transformed_results の生成は共通
        transformed_results = {}

        for parent_cluster in result_uuids_dict.values():
            parent_folder_id = parent_cluster["folder_id"]
            parent_data = parent_cluster["data"]

            if "inner" in parent_cluster:
                inner_data = {}
                for child_cluster in parent_cluster["inner"].values():
                    child_folder_id = child_cluster["folder_id"]
                    child_data = child_cluster["data"]
                    inner_data[child_folder_id] = child_data

                transformed_results[parent_folder_id] = {
                    "inner": inner_data
                }
            else:
                transformed_results[parent_folder_id] = parent_data

        if output_json:
            self._output_base_path.mkdir(parents=True, exist_ok=True)
            output_json_path = self._output_base_path / "result.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(transformed_results, f, ensure_ascii=False, indent=2)

        return transformed_results


if __name__ == "__main__":
    cl_module = InitClusteringManager(
        chroma_db=ChromaDBManager('sentence_embeddings'),
        images_folder_path='./imgs',
        output_base_path='./results'
    )
    # print(type(all_sentence_data['metadatas'][0]))
    cluster_num, _ = cl_module.get_optimal_cluster_num(embeddings=cl_module.chroma_db.get_all()['embeddings'])

    a = cl_module.chroma_db.get_all()['embeddings']
    cluster_result = cl_module.clustering(chroma_db_data=cl_module.chroma_db.get_all(), cluster_num=cluster_num,output_folder=True, output_json=True)