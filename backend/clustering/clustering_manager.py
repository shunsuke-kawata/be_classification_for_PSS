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
    
    COHESION_THRESHOLD = 0.85  # 凝集度の閾値を0.75→0.85に引き上げ（より厳しく）
    MERGE_SIMILARITY_THRESHOLD = 0.90  # クラスタ間類似度の閾値を0.85→0.90に引き上げ（より厳しく）
    
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
        """
        X-meansを使用してクラスタ数を自動決定する
        
        Args:
            embeddings: 埋め込みベクトルのリスト
            min_cluster_num: 最小クラスタ数（初期中心数として使用）
            max_cluster_num: 最大クラスタ数
        
        Returns:
            tuple[int, float]: (クラスタ数, シルエットスコア)
        """
        embeddings_np = np.array(embeddings)
        n_samples = len(embeddings_np)

        if n_samples < 3:
            print("サンプル数が少なすぎてクラスタリングできません")
            return 1, -1.0

        try:
            best_score = -1
            best_n_clusters = min_cluster_num
            
            # クラスタ数を変えながら階層型クラスタリングを試行
            for n_clusters in range(min_cluster_num, min(max_cluster_num + 1, n_samples)):
                if n_clusters < 2:
                    continue
                    
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(embeddings_np)
                
                # シルエットスコアを計算
                score = silhouette_score(embeddings_np, labels, metric='cosine')
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            
            print(f"階層型クラスタリング最適化結果: クラスタ数={best_n_clusters}, シルエットスコア={best_score:.4f}")
            
            return best_n_clusters, float(best_score)
            
        except Exception as e:
            print(f"階層型クラスタリングでエラーが発生: {e}")
            print(f"フォールバック: min_cluster_num={min_cluster_num}を使用")
            return min_cluster_num, -1.0
    
    def _calculate_cluster_cohesion(self, embeddings: list, cluster_indices: list) -> float:
        """
        クラスタ内の凝集度を計算する
        
        Args:
            embeddings: 全埋め込みベクトル
            cluster_indices: クラスタに属するインデックスのリスト
        
        Returns:
            float: 凝集度スコア（0-1、高いほど凝集度が高い）
        """
        if len(cluster_indices) <= 1:
            return 1.0
        
        cluster_embeddings = np.array([embeddings[i] for i in cluster_indices])
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # 各点からセントロイドまでのコサイン類似度の平均
        similarities = []
        for emb in cluster_embeddings:
            similarity = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            similarities.append(similarity)
        
        return float(np.mean(similarities))
    
    def _calculate_cluster_center(self, embeddings: list, cluster_indices: list) -> np.ndarray:
        """
        クラスタの中心を計算する
        
        Args:
            embeddings: 全埋め込みベクトル
            cluster_indices: クラスタに属するインデックスのリスト
        
        Returns:
            np.ndarray: クラスタの中心ベクトル
        """
        cluster_embeddings = np.array([embeddings[i] for i in cluster_indices])
        return np.mean(cluster_embeddings, axis=0)
    
    def _merge_singleton_clusters(self, clusters: dict, image_embeddings_dict: dict) -> dict:
        """
        要素数が1のクラスタを、その階層にある全ての画像特徴量に対して類似度検索を行い、
        最も類似度の高いクラスタに統合する
        
        Args:
            clusters: {cluster_id: [sentence_ids]} の辞書
            image_embeddings_dict: {sentence_id: image_embedding} の辞書
        
        Returns:
            dict: 統合後のクラスタ辞書
        """
        if len(clusters) <= 1:
            return clusters
        
        # 要素数が1のクラスタと2以上のクラスタを分離
        singleton_clusters = {}
        multi_element_clusters = {}
        
        for cluster_id, sentence_ids in clusters.items():
            if len(sentence_ids) == 1:
                singleton_clusters[cluster_id] = sentence_ids
            else:
                multi_element_clusters[cluster_id] = sentence_ids
        
        # シングルトンがない場合はそのまま返す
        if not singleton_clusters:
            return clusters
        
        # 統合先がない場合（全てがシングルトン）は最も類似度の高いペアを統合
        if not multi_element_clusters:
            print(f"  警告: 全てのクラスタが要素数1です。最も類似度の高いペアから統合します")
            return self._merge_all_singletons(singleton_clusters, image_embeddings_dict)
        
        print(f"  要素数1のクラスタ: {len(singleton_clusters)}個を統合処理")
        # 要素数1のクラスタを最も類似度の高いクラスタに統合
        merged_clusters = dict(multi_element_clusters)  # コピーを作成
        print()
        for singleton_id, sentence_ids in singleton_clusters.items():
            sentence_id = sentence_ids[0]
            
            if sentence_id not in image_embeddings_dict:
                # 画像埋め込みがない場合は統合せず個別に保持
                merged_clusters[singleton_id] = sentence_ids
                continue
            
            singleton_embedding = image_embeddings_dict[sentence_id]
            
            # その階層にある全ての画像特徴量に対して類似度検索
            best_similarity = -1
            best_cluster_id = None
            best_target_sentence_id = None
            
            # 各クラスタの全ての画像に対して類似度を計算
            for cluster_id, cluster_sentence_ids in multi_element_clusters.items():
                for target_sentence_id in cluster_sentence_ids:
                    if target_sentence_id not in image_embeddings_dict:
                        continue
                    
                    target_embedding = image_embeddings_dict[target_sentence_id]
                    
                    # コサイン類似度を計算
                    similarity = np.dot(singleton_embedding, target_embedding) / (
                        np.linalg.norm(singleton_embedding) * np.linalg.norm(target_embedding)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster_id = cluster_id
                        best_target_sentence_id = target_sentence_id
            
            # 最も類似度の高いクラスタに統合
            if best_cluster_id is not None:
                merged_clusters[best_cluster_id].append(sentence_id)
                print(f"    クラスタ{singleton_id}(要素数1)をクラスタ{best_cluster_id}に統合")
                print(f"      類似度: {best_similarity:.3f} (対象: {best_target_sentence_id})")
            else:
                # 統合先が見つからない場合は個別に保持
                merged_clusters[singleton_id] = sentence_ids
        
        return merged_clusters
    
    def _merge_all_singletons(self, singleton_clusters: dict, image_embeddings_dict: dict) -> dict:
        """
        全てのクラスタが要素数1の場合に、最も類似度の高いペアから順に統合していく
        
        Args:
            singleton_clusters: {cluster_id: [sentence_id]} の辞書（全て要素数1）
            image_embeddings_dict: {sentence_id: image_embedding} の辞書
        
        Returns:
            dict: 統合後のクラスタ辞書
        """
        # 各ペアの類似度を計算
        similarities = []
        cluster_ids = list(singleton_clusters.keys())
        
        for i, cluster_id_1 in enumerate(cluster_ids):
            sentence_id_1 = singleton_clusters[cluster_id_1][0]
            if sentence_id_1 not in image_embeddings_dict:
                continue
            
            embedding_1 = image_embeddings_dict[sentence_id_1]
            
            for j in range(i + 1, len(cluster_ids)):
                cluster_id_2 = cluster_ids[j]
                sentence_id_2 = singleton_clusters[cluster_id_2][0]
                
                if sentence_id_2 not in image_embeddings_dict:
                    continue
                
                embedding_2 = image_embeddings_dict[sentence_id_2]
                
                # コサイン類似度を計算
                similarity = np.dot(embedding_1, embedding_2) / (
                    np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)
                )
                
                similarities.append((similarity, cluster_id_1, cluster_id_2))
        
        if not similarities:
            print(f"    類似度を計算できるペアがありません。元のクラスタを返します")
            return singleton_clusters
        
        # 類似度の高い順にソート
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # 上位のペアを統合（最大で半分まで統合）
        merged_clusters = dict(singleton_clusters)
        merged_count = 0
        max_merges = max(1, len(singleton_clusters) // 2)  # 少なくとも1ペアは統合
        used_clusters = set()
        
        for similarity, cluster_id_1, cluster_id_2 in similarities:
            if merged_count >= max_merges:
                break
            
            if cluster_id_1 in used_clusters or cluster_id_2 in used_clusters:
                continue
            
            # cluster_id_2をcluster_id_1に統合
            merged_clusters[cluster_id_1].extend(merged_clusters[cluster_id_2])
            del merged_clusters[cluster_id_2]
            
            used_clusters.add(cluster_id_1)
            used_clusters.add(cluster_id_2)
            merged_count += 1
            
            print(f"    クラスタ{cluster_id_2}をクラスタ{cluster_id_1}に統合 (類似度: {similarity:.3f})")
        
        print(f"    {merged_count}ペアを統合しました")
        return merged_clusters
    
    def _merge_similar_clusters(self, clusters: dict, embeddings: list, image_embeddings_dict: dict) -> dict:
        """
        凝集度が高く、他クラスタと類似しているクラスタをマージする
        
        Args:
            clusters: {cluster_id: [sentence_ids]} の辞書
            embeddings: 文章埋め込みベクトルのリスト
            image_embeddings_dict: {sentence_id: image_embedding} の辞書
        
        Returns:
            dict: マージ後のクラスタ辞書
        """
        if len(clusters) <= 1:
            return clusters
        
        # 各クラスタの凝集度と中心を計算
        cluster_info = {}
        for cluster_id, sentence_ids in clusters.items():
            # sentence_idから対応するimage_embeddingを取得
            cluster_image_embeddings = []
            valid_indices = []
            for idx, sid in enumerate(sentence_ids):
                if sid in image_embeddings_dict:
                    cluster_image_embeddings.append(image_embeddings_dict[sid])
                    valid_indices.append(idx)
            
            if not cluster_image_embeddings:
                # 画像埋め込みがない場合でもクラスタは保持
                cluster_info[cluster_id] = {
                    'sentence_ids': sentence_ids,
                    'cohesion': 0.0,  # 凝集度なし
                    'center': None,
                    'size': len(sentence_ids)
                }
                continue
            
            # 画像特徴量での凝集度を計算
            cohesion = self._calculate_cluster_cohesion(cluster_image_embeddings, list(range(len(cluster_image_embeddings))))
            center = self._calculate_cluster_center(cluster_image_embeddings, list(range(len(cluster_image_embeddings))))
            
            cluster_info[cluster_id] = {
                'sentence_ids': sentence_ids,
                'cohesion': cohesion,
                'center': center,
                'size': len(sentence_ids)
            }
        
        # cluster_infoが空の場合は元のクラスタをそのまま返す
        if not cluster_info:
            print(f"  警告: マージ処理でcluster_infoが空です。元のクラスタを返します。")
            return clusters
        
        # マージ対象を探索
        merged = {}
        used_clusters = set()
        cluster_ids = list(cluster_info.keys())
        
        for i, cluster_id_1 in enumerate(cluster_ids):
            if cluster_id_1 in used_clusters:
                continue
            
            info_1 = cluster_info[cluster_id_1]
            
            # 凝集度が閾値未満、または中心がないクラスタは個別に保持
            if info_1['cohesion'] < self.COHESION_THRESHOLD or info_1['center'] is None:
                merged[cluster_id_1] = info_1['sentence_ids']
                used_clusters.add(cluster_id_1)
                continue
            
            # 他のクラスタとの類似度を計算
            merge_candidates = [cluster_id_1]
            
            for j, cluster_id_2 in enumerate(cluster_ids[i+1:], start=i+1):
                if cluster_id_2 in used_clusters:
                    continue
                
                info_2 = cluster_info[cluster_id_2]
                
                # 両方のクラスタが凝集度が高く、中心がある場合のみマージを検討
                if info_2['cohesion'] < self.COHESION_THRESHOLD or info_2['center'] is None:
                    continue
                
                # クラスタ中心間の類似度を計算
                similarity = np.dot(info_1['center'], info_2['center']) / (
                    np.linalg.norm(info_1['center']) * np.linalg.norm(info_2['center'])
                )
                
                if similarity >= self.MERGE_SIMILARITY_THRESHOLD:
                    merge_candidates.append(cluster_id_2)
                    used_clusters.add(cluster_id_2)
            
            # マージ実行
            if len(merge_candidates) > 1:
                print(f"  マージ: {len(merge_candidates)}個のクラスタを統合（凝集度: {info_1['cohesion']:.3f}）")
                merged_sentence_ids = []
                for cid in merge_candidates:
                    merged_sentence_ids.extend(cluster_info[cid]['sentence_ids'])
                merged[cluster_id_1] = merged_sentence_ids
            else:
                merged[cluster_id_1] = info_1['sentence_ids']
            
            used_clusters.add(cluster_id_1)
        
        # マージ結果が空の場合は元のクラスタを返す
        if not merged:
            print(f"  警告: マージ結果が空です。元のクラスタを返します。")
            return clusters
        
        return merged
    
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

    def _get_folder_name(self, captions: list[str], extra_stop_words: list[str]) -> str:
        """
        TF-IDFを使用してフォルダ名を決定する（上位3語までをカンマで連結）
        """
        if not captions:
            return Utils.generate_uuid()

        important_words = ClusteringUtils.get_tfidf_from_documents_array(
            documents=captions,
            max_words=3,
            extra_stop_words=extra_stop_words
        )

        if not important_words:
            return Utils.generate_uuid()

        # 上位語を取り出し、順序を保ったまま重複を除去
        seen = set()
        words = []
        for w, _ in important_words:
            if w and w not in seen:
                seen.add(w)
                words.append(w)

        # 最大3語までカンマで連結
        name = ",".join(words[:3])
        return name if name else Utils.generate_uuid()

    def clustering(
        self, 
        sentence_name_db_data: dict[str, list],
        image_db_data: dict[str, list],
        clustering_id_dict: dict,
        sentence_id_dict: dict,
        image_id_dict: dict,
        cluster_num: int, 
        overall_folder_name: str = None,
        output_folder: bool = False, 
        output_json: bool = False
    ):
        """
        新しい3段階クラスタリングアルゴリズム
        1. caption全体(usage + category)でクラスタリング
        2. usage + category でクラスタリング  
        3. name でクラスタリング
        各段階で凝集度ベースのマージを実行
        """
        
        print(f"=== 新3段階クラスタリングアルゴリズム開始 ===")
        print(f"Total documents: {len(sentence_name_db_data['ids'])}")
        
        # JSON出力オプションまたはフォルダ出力オプションがTrueの場合、output_base_pathをクリアして新たに作成
        if output_json or output_folder:
            if self.output_base_path.exists():
                shutil.rmtree(self.output_base_path)
                os.makedirs(self.output_base_path, exist_ok=True)
        
        # 画像埋め込みベクトルの辞書を作成（sentence_id -> image_embedding）
        image_embeddings_dict = {}
        for sentence_id in sentence_id_dict.keys():
            clustering_id = sentence_id_dict[sentence_id]['clustering_id']
            # clustering_idからimage_idを取得
            for cid, ids_dict in clustering_id_dict.items():
                if cid == clustering_id and 'image_id' in ids_dict:
                    image_id = ids_dict['image_id']
                    # image_db_dataからembeddingを取得
                    for i, iid in enumerate(image_db_data['ids']):
                        if iid == image_id:
                            image_embeddings_dict[sentence_id] = image_db_data['embeddings'][i]
                            break
                    break
        
        # ========================================
        # 第1段階: caption全体でクラスタリング (usage + category)
        # ========================================
        print("\n【第1段階】caption全体でクラスタリング")
        
        # usage + categoryの埋め込みを取得（2文目と3文目）
        usage_data = self._sentence_usage_db.get_data_by_sentence_ids(sentence_id_dict.keys())
        category_data = self._sentence_category_db.get_data_by_sentence_ids(sentence_id_dict.keys())
        
        # usage + categoryの埋め込みを結合
        combined_embeddings = []
        for i in range(len(usage_data['embeddings'])):
            combined = np.concatenate([usage_data['embeddings'][i], category_data['embeddings'][i]])
            combined_embeddings.append(combined)
        
        # クラスタ数を決定
        overall_cluster_num, _ = self.get_optimal_cluster_num(
            embeddings=combined_embeddings, 
            min_cluster_num=2, 
            max_cluster_num=min(15, len(sentence_id_dict.keys())//3)
        )
        
        print(f"  caption全体クラスタ数: {overall_cluster_num}")
        
        # クラスタリング実行
        if overall_cluster_num <= 1:
            overall_clusters = {0: list(sentence_id_dict.keys())}
        else:
            embeddings_array = np.array([emb.tolist() if hasattr(emb, 'tolist') else emb for emb in combined_embeddings])
            
            clustering = AgglomerativeClustering(
                n_clusters=overall_cluster_num,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings_array)
            
            # シルエットスコアを計算
            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(embeddings_array, labels, metric='cosine')
                print(f"  階層型クラスタリング結果: クラスタ数={overall_cluster_num}, シルエットスコア={silhouette_avg:.4f}")
            
            overall_clusters = {}
            for cluster_idx in range(overall_cluster_num):
                cluster_indices = np.where(labels == cluster_idx)[0]
                overall_clusters[cluster_idx] = [usage_data['ids'][i] for i in cluster_indices]
        
        # 凝集度ベースのマージ
        print(f"  マージ前クラスタ数: {len(overall_clusters)}")
        overall_clusters = self._merge_similar_clusters(overall_clusters, combined_embeddings, image_embeddings_dict)
        print(f"  マージ後クラスタ数: {len(overall_clusters)}")
        
        # 要素数1のクラスタを統合
        overall_clusters = self._merge_singleton_clusters(overall_clusters, image_embeddings_dict)
        print(f"  シングルトン統合後クラスタ数: {len(overall_clusters)}")
        
        # ========================================
        # 各caption全体クラスタに対して第2段階・第3段階を実行
        # ========================================
        overall_result_dict = {}
        
        for overall_idx, sentence_ids_in_overall in overall_clusters.items():
            overall_folder_id = Utils.generate_uuid()
            
            # caption全体フォルダ名を決定
            overall_captions = []
            for sentence_id in sentence_ids_in_overall:
                for i, sid in enumerate(usage_data['ids']):
                    if sid == sentence_id:
                        overall_captions.append(f"{usage_data['documents'][i].document} {category_data['documents'][i].document}")
                        break
            
            overall_folder_name_tfidf = self._get_folder_name(overall_captions, ['object','main','its','used'] + MAJOR_COLORS + MAJOR_SHAPES)
            
            print(f"\n【第2段階】usage+categoryでクラスタリング (全体クラスタ {overall_idx}: {overall_folder_name_tfidf})")
            
            # ========================================
            # 第2段階: usage + category でクラスタリング
            # ========================================
            usage_category_data = self._sentence_usage_db.get_data_by_sentence_ids(sentence_ids_in_overall)
            usage_category_cat_data = self._sentence_category_db.get_data_by_sentence_ids(sentence_ids_in_overall)
            
            # usage + categoryの埋め込みを結合
            usage_category_embeddings = []
            for i in range(len(usage_category_data['embeddings'])):
                combined = np.concatenate([usage_category_data['embeddings'][i], usage_category_cat_data['embeddings'][i]])
                usage_category_embeddings.append(combined)
            
            usage_category_cluster_num, _ = self.get_optimal_cluster_num(
                embeddings=usage_category_embeddings, 
                min_cluster_num=2, 
                max_cluster_num=min(10, len(sentence_ids_in_overall)//2)
            )
            
            print(f"  usage+categoryクラスタ数: {usage_category_cluster_num}")
            
            # クラスタリング実行
            if usage_category_cluster_num <= 1:
                usage_category_clusters = {0: sentence_ids_in_overall}
            else:
                embeddings_array = np.array([emb.tolist() if hasattr(emb, 'tolist') else emb for emb in usage_category_embeddings])
                
                clustering = AgglomerativeClustering(
                    n_clusters=usage_category_cluster_num,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(embeddings_array)
                
                # シルエットスコアを計算
                if len(set(labels)) > 1:
                    silhouette_avg = silhouette_score(embeddings_array, labels, metric='cosine')
                    print(f"  階層型クラスタリング結果: クラスタ数={usage_category_cluster_num}, シルエットスコア={silhouette_avg:.4f}")
                
                usage_category_clusters = {}
                for cluster_idx in range(usage_category_cluster_num):
                    cluster_indices = np.where(labels == cluster_idx)[0]
                    usage_category_clusters[cluster_idx] = [usage_category_data['ids'][i] for i in cluster_indices]
            
            # 凝集度ベースのマージ
            print(f"  マージ前クラスタ数: {len(usage_category_clusters)}")
            usage_category_clusters = self._merge_similar_clusters(usage_category_clusters, usage_category_embeddings, image_embeddings_dict)
            print(f"  マージ後クラスタ数: {len(usage_category_clusters)}")
            
            # 要素数1のクラスタを統合
            usage_category_clusters = self._merge_singleton_clusters(usage_category_clusters, image_embeddings_dict)
            print(f"  シングルトン統合後クラスタ数: {len(usage_category_clusters)}")
            
            # ========================================
            # 各usage+categoryクラスタに対して第3段階を実行
            # ========================================
            usage_category_result_dict = {}
            
            for usage_category_idx, sentence_ids_in_usage_category in usage_category_clusters.items():
                usage_category_folder_id = Utils.generate_uuid()
                
                # usage+categoryフォルダ名を決定
                usage_category_captions = []
                for sentence_id in sentence_ids_in_usage_category:
                    for i, sid in enumerate(usage_category_data['ids']):
                        if sid == sentence_id:
                            usage_category_captions.append(f"{usage_category_data['documents'][i].document} {usage_category_cat_data['documents'][i].document}")
                            break
                
                usage_category_folder_name = self._get_folder_name(usage_category_captions, ['object','main','its','used'] + MAJOR_COLORS + MAJOR_SHAPES)
                
                print(f"\n  【第3段階】nameでクラスタリング (usage+categoryクラスタ {usage_category_idx}: {usage_category_folder_name})")
                
                # ========================================
                # 第3段階: name でクラスタリング
                # ========================================
                name_data = self._sentence_name_db.get_data_by_sentence_ids(sentence_ids_in_usage_category)
                
                name_cluster_num, _ = self.get_optimal_cluster_num(
                    embeddings=name_data['embeddings'], 
                    min_cluster_num=2, 
                    max_cluster_num=min(10, len(sentence_ids_in_usage_category)//2)
                )
                
                print(f"    nameクラスタ数: {name_cluster_num}")
                
                # クラスタリング実行
                if name_cluster_num <= 1:
                    name_clusters = {0: sentence_ids_in_usage_category}
                else:
                    embeddings_array = np.array([emb.tolist() if hasattr(emb, 'tolist') else emb for emb in name_data['embeddings']])
                    
                    clustering = AgglomerativeClustering(
                        n_clusters=name_cluster_num,
                        metric='cosine',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(embeddings_array)
                    
                    # シルエットスコアを計算
                    if len(set(labels)) > 1:
                        silhouette_avg = silhouette_score(embeddings_array, labels, metric='cosine')
                        print(f"    階層型クラスタリング結果: クラスタ数={name_cluster_num}, シルエットスコア={silhouette_avg:.4f}")
                    
                    name_clusters = {}
                    for cluster_idx in range(name_cluster_num):
                        cluster_indices = np.where(labels == cluster_idx)[0]
                        name_clusters[cluster_idx] = [name_data['ids'][i] for i in cluster_indices]
                
                # 凝集度ベースのマージ
                print(f"    マージ前クラスタ数: {len(name_clusters)}")
                name_clusters = self._merge_similar_clusters(name_clusters, name_data['embeddings'], image_embeddings_dict)
                print(f"    マージ後クラスタ数: {len(name_clusters)}")
                
                # 要素数1のクラスタを統合
                name_clusters = self._merge_singleton_clusters(name_clusters, image_embeddings_dict)
                print(f"    シングルトン統合後クラスタ数: {len(name_clusters)}")
                
                # ========================================
                # リーフノード作成
                # ========================================
                name_result_dict = {}
                
                for name_idx, sentence_ids_in_name in name_clusters.items():
                    name_folder_id = Utils.generate_uuid()
                    
                    # 対応するclustering_idとファイルパスを取得
                    leaf_data = {}
                    leaf_captions = []
                    
                    for sentence_id in sentence_ids_in_name:
                        if sentence_id in sentence_id_dict:
                            clustering_id = sentence_id_dict[sentence_id]['clustering_id']
                            
                            # metadataを取得
                            for i, sid in enumerate(sentence_name_db_data['ids']):
                                if sid == sentence_id:
                                    leaf_data[clustering_id] = sentence_name_db_data['metadatas'][i].path
                                    leaf_captions.append(sentence_name_db_data['documents'][i].document)
                                    break
                    
                    # nameフォルダ名を決定
                    name_folder_name = self._get_folder_name(leaf_captions, ['object','main'] + MAJOR_COLORS + MAJOR_SHAPES)
                    
                    name_result_dict[name_folder_id] = {
                        'data': leaf_data,
                        'is_leaf': True,
                        'name': name_folder_name
                    }
                
                # 同じ名前のフォルダをまとめる
                name_result_dict = self._merge_folders_by_name(name_result_dict)
                
                # usage+categoryフォルダに追加
                usage_category_result_dict[usage_category_folder_id] = {
                    'data': name_result_dict,
                    'is_leaf': False,
                    'name': usage_category_folder_name
                }
            
            # 同じ名前のフォルダをまとめる
            usage_category_result_dict = self._merge_folders_by_name(usage_category_result_dict)
            
            # ⭐ 変更: caption全体フォルダは作成せず、usage+categoryフォルダを直接overall_result_dictに追加
            # 各usage+categoryフォルダをトップレベルに移動
            for folder_id, folder_data in usage_category_result_dict.items():
                overall_result_dict[folder_id] = folder_data
        
        # 同じ名前のフォルダをまとめる
        overall_result_dict = self._merge_folders_by_name(overall_result_dict)
        
        result_clustering_uuid_dict = overall_result_dict
                
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
        top_folder_id = Utils.generate_uuid()
        
        # プロジェクト名を使用するか、フォールバックとしてtop_folder_idを使用
        display_name = overall_folder_name if overall_folder_name else top_folder_id
        
        # 全体フォルダでラップしてからparent_idを追加
        wrapped_result = {
            top_folder_id: {
                "data": result_clustering_uuid_dict,
                "parent_id": None,
                "is_leaf": False,
                "name": display_name
            }
        }
        
        # 全体フォルダでラップした後にparent_idを追加
        wrapped_result = self._add_parent_ids(wrapped_result)
        
        #mongodbに登録するためのnode情報を作成する
        all_nodes,_, _ = self.create_all_nodes(wrapped_result)
        
        
        if output_json:
            # 出力ディレクトリが存在しない場合は作成
            os.makedirs(self._output_base_path, exist_ok=True)
            
            output_json_path = self._output_base_path / "result.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(wrapped_result, f, ensure_ascii=False, indent=2)  
            
            output_json_path= self._output_base_path / "all_nodes.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_nodes, f, ensure_ascii=False, indent=2)
                
        print(f"\n=== クラスタリング完了 ===")
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
                "name": folder_info.get("name",folder_id),
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