import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from chroma_db_manager import ChromaDBManager
import re
from embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from sklearn.metrics.pairwise import cosine_similarity
from utils import Utils
class InitClustering:
    
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
            raise ValueError(f" Error Folder Path: {images_folder_path}")
        
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
        best_score = -1
        best_k = min_cluster_num
        scores = []

        for k in range(min_cluster_num, max_cluster_num + 1):
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(embeddings_np)

            if len(set(labels)) == 1:
                continue 

            score = silhouette_score(embeddings_np, labels)
            scores.append((k, score))

            if score > best_score:
                best_score = score
                best_k = k

        return best_k, float(best_score)
    
    def clustering(self, chroma_db_data: dict[str, list], cluster_num: int, output_folder: bool = False, output_json: bool = False):
        

        embeddings_np = np.array(chroma_db_data['embeddings'])
        result_uuids_dict = {}

        # 親クラスタリング
        model = AgglomerativeClustering(n_clusters=cluster_num)
        labels = model.fit_predict(embeddings_np)

        if output_folder:
            if self._output_base_path.exists():
                shutil.rmtree(self._output_base_path)
            self._output_base_path.mkdir(parents=True, exist_ok=True)

        # 各親クラスタにUUIDを割り当て
        for idx in range(cluster_num):
            folder_id = Utils.generate_uuid()
            result_uuids_dict[idx] = {'folder_id': folder_id, 'ids': []}
        
        # result_uuids_dict['root_folder_id'] = Utils.generate_uuid()

        for i, label in enumerate(labels):
            result_uuids_dict[label]['ids'].append(chroma_db_data['ids'][i])

        # コサイン凝集度の計算
        def _cohesion_cosine_similarity(vectors: list[float]) -> float:
            vectors_np = np.array(vectors)
            similarity_matrix = cosine_similarity(vectors_np)
            n = len(vectors_np)
            if n < 2:
                return 1.0
            total = np.sum(similarity_matrix) - n
            return total / (n * (n - 1))

        for key, value in result_uuids_dict.items():
            folder_id = value['folder_id']
            cluster_ids = value['ids']
            chroma_db_data_in_cluster = self._chroma_db.query_by_ids(cluster_ids)

            images_embeddings = [
                ImageEmbeddingsManager.image_to_embedding(self._images_folder_path / Path(metadata.path))
                for metadata in chroma_db_data_in_cluster['metadatas']
            ]

            if len(images_embeddings) < 2:
                continue

            cohesion = _cohesion_cosine_similarity(images_embeddings)

            if cohesion >= 0.70:
                # 親クラスタとして保存（並列コピー）
                if output_folder:
                    output_dir = self._output_base_path / folder_id
                    output_dir.mkdir(parents=True, exist_ok=True)
                    Utils.copy_images_parallel(
                        chroma_db_data_in_cluster['metadatas'],
                        self._images_folder_path,
                        output_dir
                    )
                continue

            # 小クラスタリング（ネスト）
            cluster_num_in_cluster, _ = cl_module.get_optimal_cluster_num(
                embeddings=images_embeddings,
                min_cluster_num=1,
                max_cluster_num=min(len(images_embeddings), 10)
            )

            model_nested = AgglomerativeClustering(n_clusters=cluster_num_in_cluster)
            labels_nested = model_nested.fit_predict(images_embeddings)

            uuid_dict_in_cluster = {}

            for i in range(cluster_num_in_cluster):
                child_folder_id = Utils.generate_uuid()
                uuid_dict_in_cluster[i] = {'folder_id': child_folder_id, 'ids': []}

            for idx, nested_label in enumerate(labels_nested):
                uuid_dict_in_cluster[nested_label]['ids'].append(cluster_ids[idx])

            if output_folder:
                for nested_label, data in uuid_dict_in_cluster.items():
                    output_nested_dir = self._output_base_path / folder_id / data['folder_id']
                    output_nested_dir.mkdir(parents=True, exist_ok=True)

                    metadatas_to_copy = [
                        chroma_db_data_in_cluster['metadatas'][idx]
                        for idx, label in enumerate(labels_nested)
                        if label == nested_label
                    ]

                    Utils.copy_images_parallel(
                        metadatas_to_copy,
                        self._images_folder_path,
                        output_nested_dir
                    )

            result_uuids_dict[key]['inner'] = uuid_dict_in_cluster

        #出力するjsonの型を整形
        output_json = {
            "root_folder_id": Utils.generate_uuid(),
            "results": result_uuids_dict
        }
        # JSON出力
        if output_json:
            self._output_base_path.mkdir(parents=True, exist_ok=True)
            output_json_path = self._output_base_path / "result.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                
                json.dump(output_json, f, ensure_ascii=False, indent=2)

        return output_json
    
if __name__ == "__main__":
    cl_module = InitClustering(
        chroma_db=ChromaDBManager('sentence_embeddings'),
        images_folder_path='./imgs',
        output_base_path='./results'
    )
    # print(type(all_sentence_data['metadatas'][0]))
    cluster_num, _ = cl_module.get_optimal_cluster_num(embeddings=cl_module.chroma_db.get_all()['embeddings'])
    # cl_module.chroma_db.get_all()['embeddings']
    cluster_result = cl_module.clustering(chroma_db_data=cl_module.chroma_db.get_all(), cluster_num=cluster_num,output_folder=True, output_json=True)