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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from .embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from .embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from .utils import Utils
from config import DEFAULT_IMAGE_PATH,MAJOR_COLORS

class ClusteringUtils:
    @classmethod
    def get_tfidf_from_documents_array(cls, documents: list[str], max_words: int = 10, order: str = 'high',extra_stop_words:list[str]=None) -> list[tuple[str, float]]:
        """
        文書配列からTF-IDFを使用してスコアの高い/低い語を取得する
        
        Args:
            documents: 文書の配列
            max_words: 取得する最大語数（デフォルト: 10）
            order: ソート順 'high'（高い順）または 'low'（低い順）（デフォルト: 'high'）
            stop_words: ストップワード設定（デフォルト: 'english'）
        
        Returns:
            list[tuple[str, float]]: (語, スコア)のタプルの配列
        """
        if not documents or len(documents) == 0:
            return []
        
        my_stop_words = list(text.ENGLISH_STOP_WORDS.union(extra_stop_words or []))
        vectorizer = TfidfVectorizer(stop_words=my_stop_words)
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # 各語彙のスコア（文書全体での合計スコア）
        sum_scores = np.asarray(tfidf_matrix.sum(axis=0))[0]
        terms = vectorizer.get_feature_names_out()
        
        # スコア順に並べる
        if order.lower() == 'high':
            # 高い順（降順）
            sorted_idx = sum_scores.argsort()[::-1]
        elif order.lower() == 'low':
            # 低い順（昇順）
            sorted_idx = sum_scores.argsort()
        else:
            raise ValueError("order parameter must be 'high' or 'low'")
        
        # 指定された最大語数まで結果を取得
        result = []
        for i, idx in enumerate(sorted_idx[:max_words]):
            word = terms[idx]
            score = sum_scores[idx]
            result.append((word, float(score)))
        
        return result
        
        
class InitClusteringManager:
    
    COHESION_THRESHOLD = 0.75
    
    def __init__(self, sentence_name_db: ChromaDBManager, sentence_usage_db: ChromaDBManager, sentence_category_db: ChromaDBManager, image_db: ChromaDBManager, images_folder_path: str, output_base_path: str = './results'):
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
        
        self._sentence_name_db = sentence_name_db
        self._sentence_usage_db = sentence_usage_db
        self._sentence_category_db = sentence_category_db
        self._image_db = image_db
        self._images_folder_path = Path(images_folder_path)
        self._output_base_path = Path(output_base_path)
    
    @property
    def sentence_name_db(self) -> ChromaDBManager:
        return self._sentence_name_db

    @property
    def sentence_usage_db(self) -> ChromaDBManager:
        return self._sentence_usage_db

    @property
    def sentence_category_db(self) -> ChromaDBManager:
        return self._sentence_category_db

    @property
    def image_db(self)->ChromaDBManager:
        return self._image_db

    @property
    def images_folder_path(self) -> Path:
        return self._images_folder_path
    
    @property
    def output_base_path(self) -> Path:
        return self._output_base_path
    
    def register_split_document(self, document: str, chroma_sentence_id: str, metadata: 'ChromaDBManager.ChromaMetaData'):
        """
        分割された文書を3つのデータベースに登録する
        
        Args:
            document: 元の文書（3文で構成）
            chroma_sentence_id: chroma_sentence_idとなるID
            metadata: メタデータ
        """
        # 文書を3つに分割
        name_part, usage_part, category_part = ChromaDBManager.split_sentence_document(document)
        
        # 各部分のembeddingを生成
        name_embedding = SentenceEmbeddingsManager.sentence_to_embedding(name_part)
        usage_embedding = SentenceEmbeddingsManager.sentence_to_embedding(usage_part)
        category_embedding = SentenceEmbeddingsManager.sentence_to_embedding(category_part)
        
        # IDに接尾辞を付けて区別
        # related_ids = ChromaDBManager.generate_related_ids(chroma_sentence_id)
        
        # 各データベース用のメタデータを作成
        name_metadata = ChromaDBManager.ChromaMetaData(
            path=metadata.path,
            document=name_part,
            is_success=metadata.is_success,
            id=chroma_sentence_id
        )
        usage_metadata = ChromaDBManager.ChromaMetaData(
            path=metadata.path,
            document=usage_part,
            is_success=metadata.is_success,
            id=chroma_sentence_id
        )
        category_metadata = ChromaDBManager.ChromaMetaData(
            path=metadata.path,
            document=category_part,
            is_success=metadata.is_success,
            id=chroma_sentence_id
        )
        
        # 各データベースに登録
        self._sentence_name_db.add_one(name_part, name_metadata, name_embedding)
        self._sentence_usage_db.add_one(usage_part, usage_metadata, usage_embedding)
        self._sentence_category_db.add_one(category_part, category_metadata, category_embedding)
        
        return chroma_sentence_id
    
    
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
    
    def _add_parent_ids(self, clustering_dict: dict, parent_id: str = None) -> dict:
        """
        全ての要素にparent_idを追加する再帰関数
        
        Args:
            clustering_dict: クラスタリング結果の辞書
            parent_id: 親要素のID（最上位階層の場合はNone）
            
        Returns:
            parent_idが追加された辞書
        """
        result = {}
        
        for key, value in clustering_dict.items():
            # 値が辞書の場合のみ処理
            if isinstance(value, dict):
                # 現在の要素のコピーを作成
                new_value = value.copy()
                
                # parent_idを追加
                new_value['parent_id'] = parent_id
                
                # dataフィールドがある場合、再帰的に処理
                if 'data' in new_value and isinstance(new_value['data'], dict):
                    new_value['data'] = self._add_parent_ids(new_value['data'], key)
                
                result[key] = new_value
            else:
                # 文字列やその他の値の場合はそのまま
                result[key] = value
        
        return result

    def clustering(
        self, 
        sentence_name_db_data: dict[str, list],  # 変更: 明確な命名
        image_db_data: dict[str, list],
        clustering_id_dict: dict,
        sentence_id_dict: dict,  # sentence_id_dictを使用
        image_id_dict: dict,
        cluster_num: int, 
        overall_folder_name: str = None,
        output_folder: bool = False, 
        output_json: bool = False
    ):
        
        #結果を保持するdict
        result_clustering_uuid_dict = dict()
        sentence_embeddings_np = np.array(sentence_name_db_data['embeddings'])
        
        sentence_documents_data = sentence_name_db_data['documents']
        print(f"Documents count: {len(sentence_documents_data)}")
        sentence_document_data_embeddings = []
        for document in sentence_documents_data:
            sentence_document_embedding = SentenceEmbeddingsManager.sentence_to_embedding(document.document)
            sentence_document_data_embeddings.append(sentence_document_embedding)
        
        print(f"1段階目 文章特徴量でクラスタリング（名前ベース）")
        #クラスタ数が1以下の場合全てを同一のクラスタとして処理
        if(cluster_num<=1):
            folder_id = Utils.generate_uuid()
            result_clustering_uuid_dict[folder_id] = {}
            result_clustering_uuid_dict[folder_id]['data'] = {}
            
            for idx,_sentence_id in enumerate(sentence_name_db_data['ids']):

                _clustering_id = sentence_id_dict[_sentence_id]['clustering_id']
                result_clustering_uuid_dict[folder_id]['data'][_clustering_id]= sentence_name_db_data['metadatas'][idx].path

            result_clustering_uuid_dict[folder_id]['is_leaf']=True
            result_clustering_uuid_dict[folder_id]['name']=folder_id
        else:  
            # 通常通りクラスタリング
            model = AgglomerativeClustering(n_clusters=cluster_num)
            labels = model.fit_predict(sentence_embeddings_np)

            #インデックスで出力されるクラスタリング結果の格納用の一時辞書
            tmp_result_dict = dict()
            for idx in range(cluster_num):
                folder_id = Utils.generate_uuid()
                tmp_result_dict[idx] = {'folder_id': folder_id, 'data': {},'captions':{}}
            
            for i,label in enumerate(labels):
                _sentence_id = sentence_name_db_data['ids'][i]

                _clustering_id = sentence_id_dict[_sentence_id]['clustering_id']
                tmp_result_dict[label]['data'][_clustering_id] = sentence_name_db_data['metadatas'][i].path
                tmp_result_dict[label]['captions'][_clustering_id] = sentence_name_db_data['documents'][i].document

            with open('test_new.json', 'w') as f:
                json.dump(tmp_result_dict, f, indent=2)
            #結果用にjson成形
            for tmp_value_1 in tmp_result_dict.values():
                
                
                
                important_word = ClusteringUtils.get_tfidf_from_documents_array(documents=list(tmp_value_1['captions'].values()),max_words=1,extra_stop_words=['object','main']+MAJOR_COLORS)
                print(important_word)
                result_clustering_uuid_dict[tmp_value_1['folder_id']]=dict()
                result_clustering_uuid_dict[tmp_value_1['folder_id']]['data'] = tmp_value_1['data']
                result_clustering_uuid_dict[tmp_value_1['folder_id']]['is_leaf']=True
                result_clustering_uuid_dict[tmp_value_1['folder_id']]['name']=important_word[0][0]
        
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
            target_sentence_ids = []
            for _clustering_id in _clustering_id_keys:

                target_sentence_ids.append(clustering_id_dict[_clustering_id]['sentence_id'])

            # sentence_name_dbを使用して凝集度を計算
            sentence_data_in_cluster = self._sentence_name_db.get_data_by_sentence_ids(target_sentence_ids)
            
            cohesion_cosine_similarity = _cohesion_cosine_similarity(vectors=sentence_data_in_cluster['embeddings'])
            #凝集度がすでに一定以上の場合以降の処理をスキップする
            if(cohesion_cosine_similarity>self.COHESION_THRESHOLD):
                continue
            
            target_image_ids = []
            for _clustering_id in _clustering_id_keys:
                target_image_ids.append(clustering_id_dict[_clustering_id]['image_id'])

            image_data_in_cluster = self._image_db.get_data_by_ids(target_image_ids)
            
            inner_cluster_num, _ = self.get_optimal_cluster_num(embeddings=image_data_in_cluster['embeddings'],max_cluster_num=12)
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
                tmp_result_inner_dict[idx] = {'folder_id': folder_id, 'data': {},'captions':{}}

            for i,label in enumerate(labels):
                _image_id = image_data_in_cluster['ids'][i]
                
                _clustering_id = image_id_dict[_image_id]['clustering_id']
                tmp_result_inner_dict[label]['data'][_clustering_id] = image_data_in_cluster['metadatas'][i].path

            
            #結果用にjson成形
            for inner_value in tmp_result_inner_dict.values():
                result_clustering_uuid_inner_dict[inner_value['folder_id']]=dict()
                result_clustering_uuid_inner_dict[inner_value['folder_id']]['data'] = inner_value['data']
                result_clustering_uuid_inner_dict[inner_value['folder_id']]['is_leaf']=True
                result_clustering_uuid_inner_dict[inner_value['folder_id']]['name']=inner_value['folder_id']
                
            result_clustering_uuid_dict[cluster_folder_id]['data']=result_clustering_uuid_inner_dict
            result_clustering_uuid_dict[cluster_folder_id]['is_leaf']=False
            result_clustering_uuid_dict[cluster_folder_id]['name']=cluster_folder_id
        
        print(f"3段階目 文章特徴量でさらに上位階層でクラスタリング（カテゴリベース）")
        
        #上位階層クラスタリング用のdictを作成
        upper_sentence_dict = dict()

        for idx,(_folder_id,result_value) in enumerate(result_id_dict_1.items()):
            ids = list(result_value['data'].keys())

            # sentence_idを直接取得
            target_sentence_ids = []
            for id in ids:
                target_sentence_ids.append(clustering_id_dict[id]['sentence_id'])

            # sentence_category_dbを使用して上位階層のクラスタリングを行う
            upper_sentence_data = self._sentence_category_db.get_data_by_sentence_ids(target_sentence_ids)
            
            documents_data = upper_sentence_data['documents']
            upper_sentence_document_embeddings = []
            for document in documents_data:
                upper_sentence_document_embedding = SentenceEmbeddingsManager.sentence_to_embedding(document.document)
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
            upper_result_dict[idx] = {'folder_id': folder_id, 'data': {},'captions':{}}
        for idx,label in enumerate(labels):
            upper_result_dict[label]['data'].append(upper_sentence_dict[idx]['folder_id'])
        
        upper_result_clustering_uuid_dict = {value["folder_id"]: {"is_leaf": False, "data": {}, "name": value["folder_id"]}for value in upper_result_dict.values()}
        for value in upper_result_dict.values():
            upper_folder_id = value['folder_id']
            
            #分類されている子フォルダが一つの場合内部のデータを上の階層に持ってきて階層を一つ減らす
            if(len(value['data'])==1):
                upper_result_clustering_uuid_dict[upper_folder_id]['data']=result_clustering_uuid_dict[value['data'][0]]['data']
                upper_result_clustering_uuid_dict[upper_folder_id]['is_leaf']=result_clustering_uuid_dict[value['data'][0]]['is_leaf']
                upper_result_clustering_uuid_dict[upper_folder_id]['name']=upper_folder_id
            else:
                for _inner_folder_id in value['data']:
                    upper_result_clustering_uuid_dict[upper_folder_id]['data'][_inner_folder_id]= result_clustering_uuid_dict[_inner_folder_id]
                upper_result_clustering_uuid_dict[upper_folder_id]['name']=upper_folder_id
                
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
                
        # 全体をまとめたフォルダ要素でラップ
        overall_folder_id = Utils.generate_uuid()
        
        # プロジェクト名を使用するか、フォールバックとしてoverall_folder_idを使用
        display_name = overall_folder_name if overall_folder_name else overall_folder_id
        
        # 全体フォルダでラップしてからparent_idを追加
        wrapped_result = {
            overall_folder_id: {
                "data": upper_result_clustering_uuid_dict,
                "parent_id": None,
                "is_leaf": False,
                "name": display_name
            }
        }
        
        # 全体フォルダでラップした後にparent_idを追加
        #フロント側で表示するデータが
        wrapped_result = self._add_parent_ids(wrapped_result)
        
        #mongodbに登録するためのnode情報を作成する
        all_nodes,_, _ = self.create_all_nodes(wrapped_result)
        
        
        if output_json:
            output_json_path = self._output_base_path / "result.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(wrapped_result, f, ensure_ascii=False, indent=2)  
            
            output_json_path= self._output_base_path / "all_nodes.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_nodes, f, ensure_ascii=False, indent=2)
                
        return wrapped_result,all_nodes
    
    
    def create_folder_nodes(self,data, parent_id=None, result=None):
        """
        フォルダーノードを作成する
        
        Args:
            data: JSONデータ
            parent_id: 親フォルダーのID（最初のフォルダーはnull）
            result: 結果を格納するリスト
        
        Returns:
            list: フォルダーノードのリスト
        """
        if result is None:
            result = []
        
        for folder_id, folder_info in data.items():
            # フォルダーノードを作成
            folder_node = {
                "type": "folder",
                "id": folder_id,
                "name": folder_id,
                "parent_id": parent_id,
                "is_leaf": folder_info.get("is_leaf", False)
            }
            result.append(folder_node)
            
            # 子フォルダーが存在する場合、再帰的に処理
            if "data" in folder_info and not folder_info.get("is_leaf", False):
                self.create_folder_nodes(folder_info["data"], folder_id, result)
        
        return result

    def create_file_nodes(self, data, parent_id=None, result=None):
        """
        ファイルノードを作成する（is_leafがTrueのフォルダー内のファイル用）
        
        Args:
            data: JSONデータ
            parent_id: 親フォルダーのID（最初のフォルダーはnull）
            result: 結果を格納するリスト
        
        Returns:
            list: ファイルノードのリスト
        """
        if result is None:
            result = []
        
        for folder_id, folder_info in data.items():
            # is_leafがTrueのフォルダー内のファイルを処理
            if folder_info.get("is_leaf", False) and "data" in folder_info:
                for file_id, file_name in folder_info["data"].items():
                    # ファイルノードを作成
                    file_node = {
                        "type": "file",
                        "id": file_id,
                        "name": file_name,
                        "parent_id": folder_id,
                        "is_leaf": None
                    }
                    result.append(file_node)
            
            # 子フォルダーが存在する場合、再帰的に処理
            if "data" in folder_info and not folder_info.get("is_leaf", False):
                self.create_file_nodes(folder_info["data"], folder_id, result)
        
        return result

    def create_all_nodes(self,data):
        """
        フォルダーノードとファイルノードを作成する
        
        Args:
            data: JSONデータ
        
        Returns:
            tuple: (フォルダーノードのリスト, ファイルノードのリスト, 全ノードのリスト)
        """
        folder_nodes = self.create_folder_nodes(data)
        file_nodes = self.create_file_nodes(data)
        all_nodes = folder_nodes + file_nodes
        
        return  all_nodes,folder_nodes, file_nodes

if __name__ == "__main__":
    cl_module = InitClusteringManager(
        sentence_name_db=ChromaDBManager('sentence_name_embeddings'),
        sentence_usage_db=ChromaDBManager('sentence_usage_embeddings'),
        sentence_category_db=ChromaDBManager('sentence_category_embeddings'),
        image_db=ChromaDBManager('image_embeddings'),
        images_folder_path='./imgs',
        output_base_path='./results'
    )
    # print(type(all_sentence_data['metadatas'][0]))
    cluster_num, _ = cl_module.get_optimal_cluster_num(embeddings=cl_module.sentence_name_db.get_all()['embeddings'])

    a = cl_module.sentence_name_db.get_all()['embeddings']
    cluster_result = cl_module.clustering(
        sentence_name_db_data=cl_module.sentence_name_db.get_all(), 
        image_db_data=cl_module.image_db.get_all(),
        clustering_id_dict={}, 
        sentence_id_dict={}, 
        image_id_dict={}, 
        cluster_num=cluster_num,
        output_folder=True, 
        output_json=True
    )