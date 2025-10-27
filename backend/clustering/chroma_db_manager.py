import base64
import json
import os
import re
import sys
import uuid
import chromadb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from .embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from .utils import Utils

class ChromaDBManager:
    
    @staticmethod
    def split_sentence_document(document: str) -> tuple[str, str, str]:
        """
        文書を3つの部分に分割する
        
        Args:
            document (str): 分割する文書
            
        Returns:
            tuple[str, str, str]: (名前部分, 用途部分, カテゴリ部分)
        """
        sentences = re.findall(r'.+?\.(?=\s+[A-Z]|$)', document)
        
        # 3つの文に分割できない場合はデフォルト値を返す
        if len(sentences) < 3:
            # 不足分は空文字で埋める
            while len(sentences) < 3:
                sentences.append("")
        
        return sentences[0].strip(), sentences[1].strip(), sentences[2].strip()
    
    class ChromaDocument:
        def __init__(self,document:str):
            self._document = document
        
        @property
        def document(self)->str:
            return self._document
        
        def get_split_document(self)->list[str]:
            return re.findall(r'.+?\.(?=\s+[A-Z]|$)', self._document)
    class ChromaMetaData:
        def __init__(self, path: str, document: str, is_success: bool, sentence_id: str, id: str | None = None):
            #新規作成の場合UUIDを設定,そうでない時、引数から継承
            if id is None:
                self._id = Utils.generate_uuid()
            else:
                self._id = id
            
            self._path = path
            self._is_success = is_success
            self._document = document
            self._sentence_id = sentence_id

        @property
        def id(self) -> str:
            return self._id

        @property
        def path(self) -> str:
            return self._path

        @property
        def document(self)->str:
            return self._document

        @property
        def is_success(self) -> bool:
            return self._is_success
        
        @property
        def sentence_id(self) -> str:
            return self._sentence_id

        def to_dict(self) -> dict:
            return {
                "id": self.id,
                "path": self.path,
                "is_success": self.is_success,
                "document": self.document,
                "sentence_id": self.sentence_id
            }
    
        
    def __init__(self, colection_name:str,path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=colection_name)

    def add(
        self,
        documents: list[str],
        metadatas: list[ChromaMetaData],
        embeddings: list[list[float]] = None
    ) -> list[str]:
        # 既存のメタデータを取得して path -> id のマップを作成
        existing_metadata = {meta.path: meta.id for meta in self.get_all_metadata()}
        
        ids_to_return = []
        filtered_documents = []
        filtered_metadatas = []
        filtered_embeddings = []

        for i, meta in enumerate(metadatas):
            if meta.path in existing_metadata:
                # 既に存在する場合はそのIDを返却用に追加
                ids_to_return.append(existing_metadata[meta.path])
            else:
                # 新規追加するデータを蓄積
                ids_to_return.append(meta.id)
                filtered_documents.append(documents[i])
                filtered_metadatas.append(meta)
                if embeddings:
                    filtered_embeddings.append(embeddings[i])

        # 新規追加データがある場合のみupsert
        if filtered_documents:
            kwargs = {
                "ids": [meta.id for meta in filtered_metadatas],
                "documents": filtered_documents,
                "metadatas": [meta.to_dict() for meta in filtered_metadatas],
            }
            if embeddings is not None:
                kwargs["embeddings"] = filtered_embeddings
            self.collection.upsert(**kwargs)

        return ids_to_return
    
    def add_one(self, 
            document: str, 
            metadata: ChromaMetaData, 
            embeddings: list[float] = None) -> str:

        path = metadata.path

        # すでに存在するか確認
        existing = self.collection.get(
            where={"path": {"$eq": path}},
            include=["metadatas"]
        )

        #すでに存在する場合はそのレコードのIDを返す
        if existing["ids"]:
            return existing["ids"][0] 

        kwargs = {
            "ids": [metadata.id],
            "documents": [document],
            "metadatas": [metadata.to_dict()],
        }
        if embeddings is not None:
            kwargs["embeddings"] = [embeddings]
        self.collection.upsert(**kwargs)
        return metadata.id
    

    def get_all(self)->dict[str,list]:
        results = self.collection.get(include=["documents", "metadatas","embeddings"],limit=None)
        return results

    def get_all_metadata(self) -> list[ChromaMetaData]:
        all_data = self.get_all()
        return [self.ChromaMetaData(id=metadata['id'],path=metadata['path'],document=metadata['document'],is_success=metadata['is_success']) for metadata in all_data["metadatas"]]
    
    def get_all_documents(self) -> list[ChromaDocument]:
        all_data = self.get_all()
        return [self.ChromaDocument(document=document) for document in all_data["documents"]]
    
    def get_all_embeddings(self)->list[list[float]]:
        all_data = self.get_all()
        return all_data['embeddings']
    
    def update(self, ids:list[str], documents:list[str], metadatas:list[ChromaMetaData],embeddings:list[list[float]] = None)->None:
        
        metadata_dict = [meta.to_dict() for meta in metadatas]

        kwargs = {"ids": ids, "documents": documents, "metadatas": metadata_dict}
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.collection.upsert(**kwargs)
    
    def delete(self, ids:list[str])->None:
        self.collection.delete(ids=ids)

    def get_data_by_ids(
        self,
        ids: list[str]
    ) -> dict[str, list[str] | list[float] | list[ChromaMetaData] | list[ChromaDocument]]:
        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"]
        )
        return {
            'ids': results['ids'],
            'metadatas': [self.ChromaMetaData(id=metadata['id'], path=metadata['path'], document=metadata['document'], is_success=metadata['is_success'], sentence_id=metadata['sentence_id']) for metadata in results['metadatas']],
            'documents':[self.ChromaDocument(document=document) for document in results['documents']],
            'embeddings': results['embeddings'],
        }
    
    def get_data_by_id(
        self,
        id:str
    )-> dict[str, str | list[float] | ChromaMetaData | ChromaDocument] | None:
        
        results = self.collection.get(
            ids=[id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if len(results['ids']) !=1:
            return None

        return {
            'id':results['ids'][0],
            'metadata':results['metadatas'][0],
            'document':results['documents'][0],
            'embedding':results['embeddings'][0],
        }
    
    @staticmethod
    def create_separate_embeddings_from_document(
        document: str, 
        metadata: 'ChromaDBManager.ChromaMetaData'
    ) -> tuple[tuple[str, list[float]], tuple[str, list[float]], tuple[str, list[float]]]:
        """
        文書から3つの独立したembeddingを作成する
        
        Args:
            document (str): 元の文書
            metadata (ChromaMetaData): メタデータ
            
        Returns:
            tuple: ((name_sentence, name_embedding), (usage_sentence, usage_embedding), (category_sentence, category_embedding))
        """
        name_part, usage_part, category_part = ChromaDBManager.split_sentence_document(document)
        
        # 各部分のembeddingを生成
        name_embedding = SentenceEmbeddingsManager.sentence_to_embedding(name_part)
        usage_embedding = SentenceEmbeddingsManager.sentence_to_embedding(usage_part)
        category_embedding = SentenceEmbeddingsManager.sentence_to_embedding(category_part)
        
        return (
            (name_part, name_embedding),
            (usage_part, usage_embedding), 
            (category_part, category_embedding)
        )

    @staticmethod
    def extract_chroma_sentence_id(full_id: str) -> str:
        """
        完全なIDからchroma_sentence_idを抽出する
        
        Args:
            full_id: 完全なID（例: "doc123_name", "doc123_usage", "doc123_category"）
            
        Returns:
            str: chroma_sentence_id（例: "doc123"）
        """
        if '_name' in full_id:
            return full_id.replace('_name', '')
        elif '_usage' in full_id:
            return full_id.replace('_usage', '')
        elif '_category' in full_id:
            return full_id.replace('_category', '')
        else:
            return full_id
    
    @staticmethod
    def generate_related_ids(sentence_id: str) -> dict:
        """
        sentence_idから関連するIDを生成（統一ID方式）
        
        Args:
            sentence_id: 統一されたsentence_id
            
        Returns:
            dict: 同じIDを返す
        """
        return {
            "sentence_id": sentence_id,
            "name_id": sentence_id,
            "usage_id": sentence_id,
            "category_id": sentence_id
        }

    def get_data_by_metadata(
        self,
        key: str,
        value: str | int | float,
        ids: list[str] | None = None
    ):
        results = self.collection.query(
            where={key: value},
            include=["documents", "metadatas", "embeddings"],
            ids=ids if ids is not None else None
        )
        
        if len(results['ids']) != 1:
            return None

        return {
            'id': results['ids'][0],
            'metadata': self.ChromaMetaData(
                id=results['metadatas'][0]['id'],
                path=results['metadatas'][0]['path'],
                document=results['metadatas'][0]['document'],
                is_success=results['metadatas'][0]['is_success'],
                sentence_id=results['metadatas'][0]['sentence_id']
            ),
            'document': self.ChromaDocument(document=results['documents'][0]),
            'embedding': results['embeddings'][0],
        }
    
    def get_data_by_sentence_ids(self, sentence_ids: list[str]):
        """
        複数のsentence_idでデータを検索する
        """
        all_results = {'ids': [], 'metadatas': [], 'documents': [], 'embeddings': []}
        
        for sentence_id in sentence_ids:
            results = self.collection.get(
                where={"sentence_id": sentence_id},
                include=["documents", "metadatas", "embeddings"]
            )
            if results['ids']:
                all_results['ids'].extend(results['ids'])
                all_results['metadatas'].extend(results['metadatas'])
                all_results['documents'].extend(results['documents'])
                all_results['embeddings'].extend(results['embeddings'])
        
        return {
            'ids': all_results['ids'],
            'metadatas': [self.ChromaMetaData(id=metadata['id'], path=metadata['path'], document=metadata['document'], is_success=metadata['is_success'], sentence_id=metadata['sentence_id']) for metadata in all_results['metadatas']],
            'documents': [self.ChromaDocument(document=document) for document in all_results['documents']],
            'embeddings': all_results['embeddings'],
        }
    
    def query_by_embeddings(
        self,
        query_embeddings: list[list[float]],
        ids: list[str] | None = None,
        n_results: int = 10,
        distance_threshold: float | None = None
    ) -> dict[str, list[str] | list[ChromaMetaData] | list[ChromaDocument] | list[float]]:
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances", "embeddings"],
            ids=ids if ids is not None else None
        )
        
        result_dict = {
            'ids':results['ids'][0],
            'metadatas':[self.ChromaMetaData(id=metadata['id'],path=metadata['path'],document=metadata['document'],is_success=metadata['is_success']) for metadata in results['metadatas'][0]],
            'documents':[self.ChromaDocument(document=document)for document in results['documents'][0]],
            "embeddings":results['embeddings'][0],
            'distances':results['distances'][0],
        }
        

        # distance_threshold が指定されていない場合はそのまま返す
        if distance_threshold is not None:
            filtered_result = {
                'ids': [],
                'metadatas': [],
                'documents': [],
                'embeddings': [],
                'distances': [],
            }

            for _id, meta, doc, emb, dist in zip(results['ids'][0],results['metadatas'][0],results['documents'][0],results['embeddings'][0],results['distances'][0]):
                meta_obj = self.ChromaMetaData(id=meta['id'],path=meta['path'],document=meta['document'],is_success=meta['is_success'])
                if dist <= distance_threshold:
                    filtered_result['ids'].append(_id)
                    filtered_result['metadatas'].append(meta_obj)
                    filtered_result['documents'].append(doc)
                    filtered_result['embeddings'].append(emb)
                    filtered_result['distances'].append(dist)
            return filtered_result
                
        return result_dict
 

if __name__ == "__main__":


    # Sentence Embedding（テキストベース）- 3つのデータベースに分離
    sentence_name_db_manager = ChromaDBManager("sentence_name_embeddings")
    sentence_usage_db_manager = ChromaDBManager("sentence_usage_embeddings")
    sentence_category_db_manager = ChromaDBManager("sentence_category_embeddings")
    
    # テスト用（name DBを使用）
    _ = sentence_name_db_manager.get_all()
    