import json
import os
from pathlib import Path
import shutil
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from .chroma_db_manager import ChromaDBManager
import re
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from .embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from .utils import Utils
from config import DEFAULT_IMAGE_PATH
class InitClusteringManager:
    
    COHESION_THRESHOLD = 0.75
    
    def __init__(self, sentence_db: ChromaDBManager, image_db:ChromaDBManager,images_folder_path: str, output_base_path: str = './results'):
        def _is_valid_path(path: str) -> bool:
            if not isinstance(path, str) or not path.strip():
                return False

            if not (path.startswith("./") or path.startswith("../") or path.startswith("/")):
                return False

            if path.endswith("/"):
                return False

            if re.search(r'[<>:"|?*]', path):
                return False

            return True

        if not (_is_valid_path(images_folder_path) and _is_valid_path(output_base_path)):
            raise ValueError(f" Error Folder Path: {images_folder_path}, {output_base_path}")
        
        self._sentence_db = sentence_db
        self._image_db = image_db
        self._images_folder_path = Path(images_folder_path)
        self._output_base_path = Path(output_base_path)
    
    @property
    def sentence_db(self) -> ChromaDBManager:
        return self._sentence_db

    @property
    def image_db(self)->ChromaDBManager:
        return self._image_db

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
        
        return best_k, float(best_score) if best_score >= 0 else (1, -1.0)
    
    def clustering(
        self, 
        sentence_db_data: dict[str, list], 
        image_db_data:dict[str,list],
        clustering_id_dict:dict,
        sentence_id_dict:dict,
        image_id_dict:dict,
        cluster_num: int, 
        output_folder: bool = False, 
        output_json: bool = False
    ):
        
        #結果を保持するdict
        result_clustering_uuid_dict = dict()
        sentence_embeddings_np = np.array(sentence_db_data['embeddings'])
        
        print(f"1段階目 文章特徴量でクラスタリング")
        #クラスタ数が1以下の場合全てを同一のクラスタとして処理
        if(cluster_num<=1):
            folder_id = Utils.generate_uuid()
            result_clustering_uuid_dict[0] = {folder_id:{}}
            
            for idx,_sentence_id in enumerate(sentence_db_data['ids']):
                _clustering_id = sentence_id_dict[_sentence_id]['clustering_id']
                result_clustering_uuid_dict[folder_id]['data'][_clustering_id]= sentence_db_data['metadatas'][idx].path
                result_clustering_uuid_dict[folder_id]['is_leaf']=True
        else:  
            # 通常通りクラスタリング
            model = AgglomerativeClustering(n_clusters=cluster_num)
            labels = model.fit_predict(sentence_embeddings_np)

            #インデックスで出力されるクラスタリング結果の格納用の一時辞書
            tmp_result_dict = dict()
            for idx in range(cluster_num):
                folder_id = Utils.generate_uuid()
                tmp_result_dict[idx] = {'folder_id': folder_id, 'data': {}}
            
            for i,label in enumerate(labels):
                _clustering_id = sentence_id_dict[sentence_db_data['ids'][i]]['clustering_id']
                tmp_result_dict[label]['data'][_clustering_id] = sentence_db_data['metadatas'][i].path
                
            #結果用にjson成形
            for value in tmp_result_dict.values():
            
                result_clustering_uuid_dict[value['folder_id']]=dict()
                result_clustering_uuid_dict[value['folder_id']]['data'] = value['data']
                result_clustering_uuid_dict[value['folder_id']]['is_leaf']=True
        
        print(f"2段階目 凝集度が低いものを画像特徴量でクラスタリング")
        #コサイン類似度で凝集度を判定する関数
        def _cohesion_cosine_similarity(vectors: list[float]) -> float:
            vectors_np = np.array(vectors)
            similarity_matrix = cosine_similarity(vectors_np)
            n = len(vectors_np)
            if n < 2:
                return 1.0
            total = np.sum(similarity_matrix) - n
            return total / (n * (n - 1))
        
        #1階層目のresult_uuid_dictを保存
        result_id_dict_1 = copy.deepcopy(result_clustering_uuid_dict)
        
        for cluster_folder_id,value in result_clustering_uuid_dict.items():
            
            result_clustering_uuid_inner_dict = dict()
            _clustering_id_keys = list(value['data'].keys())
            
            #クラスタ内の画像の文章データベースのIDを抽出
            target_sentence_ids = [clustering_id_dict[_clustering_id]['sentence_id'] for _clustering_id in _clustering_id_keys]

            sentence_data_in_cluster = self._sentence_db.get_data_by_ids(ids=target_sentence_ids)
            
            cohesion_cosine_similarity = _cohesion_cosine_similarity(vectors=sentence_data_in_cluster['embeddings'])
            #凝集度がすでに一定以上の場合以降の処理をスキップする
            if(cohesion_cosine_similarity>self.COHESION_THRESHOLD):
                continue
            
            target_image_ids = [clustering_id_dict[_clustering_id]['image_id'] for _clustering_id in _clustering_id_keys]

            image_data_in_cluster = self._image_db.get_data_by_ids(target_image_ids)
            
            inner_cluster_num, _ = self.get_optimal_cluster_num(embeddings=image_data_in_cluster['embeddings'],max_cluster_num=12)
            print(f"  {cluster_folder_id} 画像特徴量クラスタ数：{inner_cluster_num}")
            #結果としてクラスタ数が1以下の場合もスキップする
            if inner_cluster_num <=1:
                continue
            
            image_embeddings_np = np.array(image_data_in_cluster['embeddings'])
            model = AgglomerativeClustering(n_clusters=inner_cluster_num)
            labels = model.fit_predict(image_embeddings_np)
            
            #インデックスで出力されるクラスタリング結果の格納用の一時辞書
            tmp_result_inner_dict = dict()
            for idx in range(inner_cluster_num):
                folder_id = Utils.generate_uuid()
                tmp_result_inner_dict[idx] = {'folder_id': folder_id, 'data': {}}

            for i,label in enumerate(labels):
                _clustering_id = image_id_dict[image_data_in_cluster['ids'][i]]['clustering_id']
                tmp_result_inner_dict[label]['data'][_clustering_id] = image_data_in_cluster['metadatas'][i].path
            
            #結果用にjson成形
            for inner_value in tmp_result_inner_dict.values():
                result_clustering_uuid_inner_dict[inner_value['folder_id']]=dict()
                result_clustering_uuid_inner_dict[inner_value['folder_id']]['data'] = inner_value['data']
                result_clustering_uuid_inner_dict[inner_value['folder_id']]['is_leaf']=True
                
            result_clustering_uuid_dict[cluster_folder_id]['data']=result_clustering_uuid_inner_dict
            result_clustering_uuid_dict[cluster_folder_id]['is_leaf']=False
        
        print(f"3段階目 文章特徴量でさらに上位階層でクラスタリング")
        
        #上位階層クラスタリング用のdictを作成
        upper_sentence_dict = dict()

        for idx,(_folder_id,result_value) in enumerate(result_id_dict_1.items()):
            ids = list(result_value['data'].keys())

            target_sentence_ids = [clustering_id_dict[id]['sentence_id'] for id in ids]
            upper_sentence_data = self._sentence_db.get_data_by_ids(ids=target_sentence_ids)
            
            documents_data = upper_sentence_data['documents']
            upper_sentence_document_embeddings = []
            for document in documents_data:
                upper_sentence_document_embedding = SentenceEmbeddingsManager.sentence_to_embedding(''.join(document.get_split_document()[1:]))
                upper_sentence_document_embeddings.append(upper_sentence_document_embedding)
            
            upper_embeddings_np = np.array(upper_sentence_document_embeddings)
             
            ave_upper_embeddings = np.average(upper_embeddings_np,axis=0).tolist()
            upper_sentence_dict[idx]={}
            upper_sentence_dict[idx]['embeddings']=ave_upper_embeddings
            upper_sentence_dict[idx]['folder_id']=_folder_id
        
        upper_cluster_num,_ = self.get_optimal_cluster_num(np.array([v['embeddings'] for v in upper_sentence_dict.values()]),min_cluster_num=2,max_cluster_num=int(len(list(upper_sentence_dict.values()))/2))
        print(f"上位階層クラスタ数：{upper_cluster_num}")
        model = AgglomerativeClustering(n_clusters=upper_cluster_num)
        labels = model.fit_predict(np.array([v['embeddings'] for v in upper_sentence_dict.values()]))
        
        #結果のクラスタ用のdict
        upper_result_dict = dict()
        for idx in range(upper_cluster_num):
            folder_id = Utils.generate_uuid()
            upper_result_dict[idx] = {'folder_id': folder_id, 'data': []}
        for idx,label in enumerate(labels):
            upper_result_dict[label]['data'].append(upper_sentence_dict[idx]['folder_id'])
        
        upper_result_clustering_uuid_dict = {value["folder_id"]: {"is_leaf": False, "data": {}}for value in upper_result_dict.values()}
        for value in upper_result_dict.values():
            upper_folder_id = value['folder_id']
            
            #分類されている子フォルダが一つの場合内部のデータを上の階層に持ってきて階層を一つ減らす
            if(len(value['data'])==1):
                upper_result_clustering_uuid_dict[upper_folder_id]['data']=result_clustering_uuid_dict[value['data'][0]]['data']
                upper_result_clustering_uuid_dict[upper_folder_id]['is_leaf']=result_clustering_uuid_dict[value['data'][0]]['is_leaf']
            else:
                for _inner_folder_id in value['data']:
                    upper_result_clustering_uuid_dict[upper_folder_id]['data'][_inner_folder_id]= result_clustering_uuid_dict[_inner_folder_id]
                
        # JSON出力オプションまたはフォルダ出力オプションがTrueの場合、output_base_pathをクリアして新たに作成
        if output_json or output_folder:
            if self.output_base_path.exists():
                shutil.rmtree(self.output_base_path)
                os.makedirs(self.output_base_path, exist_ok=True)
                
        #フォルダ出力オプションがTrueの時クラスタリング結果をフォルダとして出力
        if output_folder:
            def copy_tree(node: dict, output_path: Path, images_folder_path: Path):
                os.makedirs(output_path, exist_ok=True)
                
                if node['is_leaf']:
                    for filename in node['data'].values():
                        src = images_folder_path / Path(filename)
                        dst = output_path / Path(filename)
                        shutil.copy(src, dst)
                else:
                    for folder_id, child_node in node['data'].items():
                        child_output_path = output_path / Path(folder_id)
                        copy_tree(child_node, child_output_path, images_folder_path)
                        
            for _folder_id, value in upper_result_clustering_uuid_dict.items():
                output_path = self.output_base_path / Path(_folder_id)
                copy_tree(value, output_path, self.images_folder_path)
                
        if output_json:
            output_json_path = self._output_base_path / "result.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(upper_result_clustering_uuid_dict, f, ensure_ascii=False, indent=2)        
        return upper_result_clustering_uuid_dict

if __name__ == "__main__":
    cl_module = InitClusteringManager(
        chroma_db=ChromaDBManager('sentence_embeddings'),
        images_folder_path='./imgs',
        output_base_path='./results'
    )
    # print(type(all_sentence_data['metadatas'][0]))
    cluster_num, _ = cl_module.get_optimal_cluster_num(embeddings=cl_module.sentence_db.get_all()['embeddings'])

    a = cl_module.chroma_db.get_all()['embeddings']
    cluster_result = cl_module.clustering(chroma_db_data=cl_module.sentence_db.get_all(), cluster_num=cluster_num,output_folder=True, output_json=True)