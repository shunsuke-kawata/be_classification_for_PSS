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
from config import DEFAULT_IMAGE_PATH,MAJOR_COLORS, MAJOR_SHAPES

class ClusteringUtils:
    @classmethod
    def get_tfidf_from_documents_array(cls, documents: list[str], max_words: int = 10, order: str = 'high',extra_stop_words:list[str]=None) -> list[tuple[str, float]]:
        """
        文書配列からTF-IDFを使用してスコアの高い/低い語を取得する
        空の語彙エラーが発生した場合、段階的にストップワードを減らして再試行する
        
        Args:
            documents: 文書の配列
            max_words: 取得する最大語数（デフォルト: 10）
            order: ソート順 'high'（高い順）または 'low'（低い順）（デフォルト: 'high'）
            extra_stop_words: 追加ストップワード設定
        
        Returns:
            list[tuple[str, float]]: (語, スコア)のタプルの配列
        """
        if not documents or len(documents) == 0:
            return []
        
        # ベースのストップワードを設定
        base_stop_words = list(text.ENGLISH_STOP_WORDS)
        extra_stop_words = extra_stop_words or []
        
        # 段階的にストップワードを減らして試行
        stop_word_configurations = [
            # 1回目: 全てのストップワードを使用
            base_stop_words + extra_stop_words,
            # 2回目: MAJOR_COLORSを除外
            base_stop_words + [word for word in extra_stop_words if word not in MAJOR_COLORS],
            # 3回目: 'shape'も除外
            base_stop_words + [word for word in extra_stop_words if word not in MAJOR_COLORS and word != 'shape'],
            # 4回目: ベースのストップワードのみ
            base_stop_words,
            # 5回目: ストップワードなし
            None
        ]
        
        for attempt, stop_words in enumerate(stop_word_configurations, 1):
            try:
                vectorizer = TfidfVectorizer(stop_words=stop_words)
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
                
                if attempt > 1:
                    print(f"TF-IDF成功: {attempt}回目の試行でストップワード設定を調整しました")
                
                return result
                
            except ValueError as e:
                if "empty vocabulary" in str(e):
                    print(f"TF-IDF試行 {attempt}: 空の語彙エラー - ストップワードを調整して再試行")
                    continue
                else:
                    raise e
        
        # 全ての試行が失敗した場合
        print("警告: TF-IDFで語彙を取得できませんでした。空のリストを返します。")
        return []
        
        
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
    
    def _merge_folders_by_name(self, folder_dict: dict) -> dict:
        """
        同じ名前のフォルダをまとめる
        
        Args:
            folder_dict: フォルダの辞書 {folder_id: {data, is_leaf, name}}
            
        Returns:
            dict: 同じ名前のフォルダをまとめた辞書
        """
        if not folder_dict:
            return folder_dict
            
        # 名前ごとにフォルダをグループ化
        name_groups = {}
        for folder_id, folder_info in folder_dict.items():
            folder_name = folder_info.get('name', folder_id)
            if folder_name not in name_groups:
                name_groups[folder_name] = []
            name_groups[folder_name].append((folder_id, folder_info))
        
        # 名前が重複していないフォルダはそのまま、重複しているフォルダはまとめる
        merged_dict = {}
        for folder_name, folder_list in name_groups.items():
            if len(folder_list) == 1:
                # 重複なし：そのまま追加
                folder_id, folder_info = folder_list[0]
                merged_dict[folder_id] = folder_info
            else:
                # 重複あり：まとめる
                print(f"同じ名前 '{folder_name}' のフォルダを {len(folder_list)} 個まとめます")
                
                # 新しいフォルダIDを生成
                merged_folder_id = Utils.generate_uuid()
                merged_data = {}
                all_is_leaf = True
                
                for folder_id, folder_info in folder_list:
                    if folder_info.get('is_leaf', False):
                        # リーフノードの場合：dataを直接マージ
                        if isinstance(folder_info.get('data'), dict):
                            merged_data.update(folder_info['data'])
                    else:
                        # 非リーフノードの場合：子フォルダをマージ
                        all_is_leaf = False
                        if isinstance(folder_info.get('data'), dict):
                            merged_data.update(folder_info['data'])
                
                merged_dict[merged_folder_id] = {
                    'data': merged_data,
                    'is_leaf': all_is_leaf,
                    'name': folder_name
                }
        
        return merged_dict

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
        
        print(f"Usage → Name 2段階クラスタリングアルゴリズム開始")
        print(f"Total documents: {len(sentence_name_db_data['ids'])}")
        
        # JSON出力オプションまたはフォルダ出力オプションがTrueの場合、output_base_pathをクリアして新たに作成
        if output_json or output_folder:
            if self.output_base_path.exists():
                shutil.rmtree(self.output_base_path)
                os.makedirs(self.output_base_path, exist_ok=True)
        
        print("\n1段階目: Usage DB でクラスタリング")
        # Usage DBからデータを取得
        usage_data = self._sentence_usage_db.get_data_by_sentence_ids(sentence_id_dict.keys())
        usage_cluster_num, _ = self.get_optimal_cluster_num(
            embeddings=usage_data['embeddings'], 
            min_cluster_num=2, 
            max_cluster_num=min(10, len(sentence_id_dict.keys())//2)
        )
        
        print(f"Usage クラスタ数: {usage_cluster_num}")
        
        if usage_cluster_num <= 1:
            print("Usage クラスタ数が1以下のため、全体をName DBで直接クラスタリング")
            # Usage で分類できない場合は、Name DBで直接クラスタリング
            name_data = self._sentence_name_db.get_data_by_sentence_ids(sentence_id_dict.keys())
            name_cluster_num, _ = self.get_optimal_cluster_num(
                embeddings=name_data['embeddings'], 
                min_cluster_num=2, 
                max_cluster_num=min(10, len(sentence_id_dict.keys())//2)
            )
            
            if name_cluster_num <= 1:
                print("Name クラスタ数も1以下のため、全体を1つのリーフノードとして処理")
                # 単一クラスタの場合、リーフノードとして処理
                
                # 対応するclustering_idを取得
                clustering_ids_all = []
                for sentence_id in sentence_id_dict.keys():
                    if sentence_id in sentence_id_dict:
                        clustering_ids_all.append(sentence_id_dict[sentence_id]['clustering_id'])
                
                # リーフノードとして処理
                leaf_data = {}
                leaf_captions = []
                for clustering_id in clustering_ids_all:
                    # sentence_idからclustering_idに対応するmetadataを取得
                    sentence_id = None
                    for cid, ids_dict in clustering_id_dict.items():
                        if cid == clustering_id:
                            sentence_id = ids_dict['sentence_id']
                            break
                    
                    if sentence_id:
                        # sentence_name_db_dataから該当するmetadataを取得
                        for i, sid in enumerate(sentence_name_db_data['ids']):
                            if sid == sentence_id:
                                leaf_data[clustering_id] = sentence_name_db_data['metadatas'][i].path
                                leaf_captions.append(sentence_name_db_data['documents'][i].document)
                                break
                
                # TF-IDFで名前を決定
                if leaf_captions:
                    important_word = ClusteringUtils.get_tfidf_from_documents_array(
                        documents=leaf_captions,
                        max_words=1,
                        extra_stop_words=['object','main'] + MAJOR_COLORS + MAJOR_SHAPES
                    )
                    folder_name = important_word[0][0] if important_word else Utils.generate_uuid()
                else:
                    folder_name = Utils.generate_uuid()
                
                main_folder_id = Utils.generate_uuid()
                result_dict = {
                    main_folder_id: {
                        'data': leaf_data,
                        'is_leaf': True,
                        'name': folder_name
                    }
                }
            else:
                # Name でクラスタリング実行
                model = AgglomerativeClustering(n_clusters=name_cluster_num)
                labels = model.fit_predict(np.array(name_data['embeddings']))
                
                # クラスタごとにsentence_idを分類
                name_clusters = {}
                for i, label in enumerate(labels):
                    if label not in name_clusters:
                        name_clusters[label] = []
                    name_clusters[label].append(name_data['ids'][i])
                
                # Name レベルの結果を構築
                result_dict = {}
                for name_idx, sentence_ids_in_name in name_clusters.items():
                    name_folder_id = Utils.generate_uuid()
                    
                    # 対応するclustering_idを取得
                    clustering_ids_in_name = []
                    for sentence_id in sentence_ids_in_name:
                        if sentence_id in sentence_id_dict:
                            clustering_ids_in_name.append(sentence_id_dict[sentence_id]['clustering_id'])
                    
                    # リーフノードとして処理
                    leaf_data = {}
                    leaf_captions = []
                    for clustering_id in clustering_ids_in_name:
                        # sentence_idからclustering_idに対応するmetadataを取得
                        sentence_id = None
                        for cid, ids_dict in clustering_id_dict.items():
                            if cid == clustering_id:
                                sentence_id = ids_dict['sentence_id']
                                break
                        
                        if sentence_id:
                            # sentence_name_db_dataから該当するmetadataを取得
                            for i, sid in enumerate(sentence_name_db_data['ids']):
                                if sid == sentence_id:
                                    leaf_data[clustering_id] = sentence_name_db_data['metadatas'][i].path
                                    leaf_captions.append(sentence_name_db_data['documents'][i].document)
                                    break
                    
                    # TF-IDFで名前を決定
                    if leaf_captions:
                        important_word = ClusteringUtils.get_tfidf_from_documents_array(
                            documents=leaf_captions,
                            max_words=1,
                            extra_stop_words=['object','main'] + MAJOR_COLORS + MAJOR_SHAPES
                        )
                        folder_name = important_word[0][0] if important_word else name_folder_id
                    else:
                        folder_name = name_folder_id
                    
                    result_dict[name_folder_id] = {
                        'data': leaf_data,
                        'is_leaf': True,
                        'name': folder_name
                    }
        else:
            # Usage でクラスタリング実行
            model = AgglomerativeClustering(n_clusters=usage_cluster_num)
            labels = model.fit_predict(np.array(usage_data['embeddings']))
            
            # クラスタごとにsentence_idを分類
            usage_clusters = {}
            for i, label in enumerate(labels):
                if label not in usage_clusters:
                    usage_clusters[label] = []
                usage_clusters[label].append(usage_data['ids'][i])
            
            # Usage レベルの結果を構築
            result_dict = {}
            
            for usage_idx, sentence_ids_in_usage in usage_clusters.items():
                usage_folder_id = Utils.generate_uuid()
                
                # Usage用のTF-IDF名前決定
                usage_captions = []
                for sentence_id in sentence_ids_in_usage:
                    for i, sid in enumerate(usage_data['ids']):
                        if sid == sentence_id:
                            usage_captions.append(usage_data['documents'][i].document)
                            break
                
                if usage_captions:
                    usage_important_word = ClusteringUtils.get_tfidf_from_documents_array(
                        documents=usage_captions,
                        max_words=1,
                        extra_stop_words=['object','main','its','used'] + MAJOR_COLORS + MAJOR_SHAPES
                    )
                    usage_folder_name = usage_important_word[0][0] if usage_important_word else usage_folder_id
                else:
                    usage_folder_name = usage_folder_id
                
                print(f"\n2段階目: Name DB でクラスタリング (Usage {usage_idx}: {usage_folder_name})")
                # Name DBからデータを取得
                name_data = self._sentence_name_db.get_data_by_sentence_ids(sentence_ids_in_usage)
                name_cluster_num, _ = self.get_optimal_cluster_num(
                    embeddings=name_data['embeddings'], 
                    min_cluster_num=2, 
                    max_cluster_num=min(10, len(sentence_ids_in_usage)//2)
                )
                
                print(f"Name クラスタ数: {name_cluster_num}")
                
                if name_cluster_num <= 1:
                    print("Name クラスタ数が1以下のため、リーフノードとして処理")
                    # 単一クラスタの場合、リーフノードとして処理
                    
                    # 対応するclustering_idを取得
                    clustering_ids_in_usage = []
                    for sentence_id in sentence_ids_in_usage:
                        if sentence_id in sentence_id_dict:
                            clustering_ids_in_usage.append(sentence_id_dict[sentence_id]['clustering_id'])
                    
                    # リーフノードとして処理
                    leaf_data = {}
                    leaf_captions = []
                    for clustering_id in clustering_ids_in_usage:
                        # sentence_idからclustering_idに対応するmetadataを取得
                        sentence_id = None
                        for cid, ids_dict in clustering_id_dict.items():
                            if cid == clustering_id:
                                sentence_id = ids_dict['sentence_id']
                                break
                        
                        if sentence_id:
                            # sentence_name_db_dataから該当するmetadataを取得
                            for i, sid in enumerate(sentence_name_db_data['ids']):
                                if sid == sentence_id:
                                    leaf_data[clustering_id] = sentence_name_db_data['metadatas'][i].path
                                    leaf_captions.append(sentence_name_db_data['documents'][i].document)
                                    break
                    
                    # TF-IDFで名前を決定
                    if leaf_captions:
                        important_word = ClusteringUtils.get_tfidf_from_documents_array(
                            documents=leaf_captions,
                            max_words=1,
                            extra_stop_words=['object','main'] + MAJOR_COLORS + MAJOR_SHAPES
                        )
                        folder_name = important_word[0][0] if important_word else usage_folder_name
                    else:
                        folder_name = usage_folder_name
                    
                    result_dict[usage_folder_id] = {
                        'data': leaf_data,
                        'is_leaf': True,
                        'name': folder_name
                    }
                else:
                    # Name でクラスタリング実行
                    model = AgglomerativeClustering(n_clusters=name_cluster_num)
                    labels = model.fit_predict(np.array(name_data['embeddings']))
                    
                    # クラスタごとにsentence_idを分類
                    name_clusters = {}
                    for i, label in enumerate(labels):
                        if label not in name_clusters:
                            name_clusters[label] = []
                        name_clusters[label].append(name_data['ids'][i])
                    
                    # Name レベルの結果を構築
                    name_result_dict = {}
                    
                    for name_idx, sentence_ids_in_name in name_clusters.items():
                        name_folder_id = Utils.generate_uuid()
                        
                        # 対応するclustering_idを取得
                        clustering_ids_in_name = []
                        for sentence_id in sentence_ids_in_name:
                            if sentence_id in sentence_id_dict:
                                clustering_ids_in_name.append(sentence_id_dict[sentence_id]['clustering_id'])
                        
                        # リーフノードとして処理
                        leaf_data = {}
                        leaf_captions = []
                        for clustering_id in clustering_ids_in_name:
                            # sentence_idからclustering_idに対応するmetadataを取得
                            sentence_id = None
                            for cid, ids_dict in clustering_id_dict.items():
                                if cid == clustering_id:
                                    sentence_id = ids_dict['sentence_id']
                                    break
                            
                            if sentence_id:
                                # sentence_name_db_dataから該当するmetadataを取得
                                for i, sid in enumerate(sentence_name_db_data['ids']):
                                    if sid == sentence_id:
                                        leaf_data[clustering_id] = sentence_name_db_data['metadatas'][i].path
                                        leaf_captions.append(sentence_name_db_data['documents'][i].document)
                                        break
                        
                        # TF-IDFで名前を決定
                        if leaf_captions:
                            important_word = ClusteringUtils.get_tfidf_from_documents_array(
                                documents=leaf_captions,
                                max_words=1,
                                extra_stop_words=['object','main'] + MAJOR_COLORS + MAJOR_SHAPES
                            )
                            folder_name = important_word[0][0] if important_word else name_folder_id
                        else:
                            folder_name = name_folder_id
                        
                        name_result_dict[name_folder_id] = {
                            'data': leaf_data,
                            'is_leaf': True,
                            'name': folder_name
                        }
                    
                    # Name段階で同じ名前のフォルダをまとめる
                    name_result_dict = self._merge_folders_by_name(name_result_dict)
                    
                    result_dict[usage_folder_id] = {
                        'data': name_result_dict,
                        'is_leaf': False,
                        'name': usage_folder_name
                    }
        
        # Usage段階で同じ名前のフォルダをまとめる
        result_dict = self._merge_folders_by_name(result_dict)
        
        # 最終的な結果を設定
        result_clustering_uuid_dict = result_dict
                
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
                        
            for _folder_id, value in result_clustering_uuid_dict.items():
                output_path = self.output_base_path / Path(_folder_id)
                copy_tree(value, output_path, self.images_folder_path)
                
        # 全体をまとめたフォルダ要素でラップ
        overall_folder_id = Utils.generate_uuid()
        
        # プロジェクト名を使用するか、フォールバックとしてoverall_folder_idを使用
        display_name = overall_folder_name if overall_folder_name else overall_folder_id
        
        # 全体フォルダでラップしてからparent_idを追加
        wrapped_result = {
            overall_folder_id: {
                "data": result_clustering_uuid_dict,
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