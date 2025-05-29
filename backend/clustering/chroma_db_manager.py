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
    
    class ChromaDocument:
        def __init__(self,document:str):
            self._document = document
        
        @property
        def document(self)->str:
            return self._document
        
        def get_split_document(self)->list[str]:
            return re.findall(r'.+?\.(?=\s+[A-Z]|$)', self._document)
    class ChromaMetaData:
        def __init__(self, path: str,document:str,is_success: bool,id:str | None=None):
            #新規作成の場合UUIDを設定,そうでない時、引数から継承
            if id is None:
                self._id = Utils.generate_uuid()
            else:
                self._id = id
            
            self._path = path
            self._is_success = is_success
            self._document = document

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

        def to_dict(self) -> dict:
            return {
                "id": self.id,
                "path": self.path,
                "is_success": self.is_success,
                "document":self.document
            }
    
        
    def __init__(self, colection_name:str,path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=colection_name)

    def add(self,documents:list[str], metadatas:list[ChromaMetaData],embeddings:list[list[float]] = None)->None:
        existing_paths = {meta.path for meta in self.get_all_metadata()}

        filtered_indices = [i for i, meta in enumerate(metadatas) if meta.path not in existing_paths]

        if not filtered_indices:
            return

        # フィルター後のデータを抽出
        filtered_documents = [documents[i] for i in filtered_indices]
        filtered_metadatas = [metadatas[i] for i in filtered_indices]

        kwargs = {
            "ids": [meta.id for meta in filtered_metadatas],
            "documents": filtered_documents,
            "metadatas": [meta.to_dict() for meta in filtered_metadatas],
        }
        filtered_embeddings = [embeddings[i] for i in filtered_indices] if embeddings else None
        if filtered_embeddings:
            kwargs["embeddings"] = filtered_embeddings

        self.collection.upsert(**kwargs)

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
        return all_data
    
    def update(self, ids:list[str], documents:list[str], metadatas:list[ChromaMetaData],embeddings:list[list[float]] = None)->None:
        
        metadata_dict = [meta.to_dict() for meta in metadatas]

        kwargs = {"ids": ids, "documents": documents, "metadatas": metadata_dict}
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.collection.upsert(**kwargs)
    
    def delete(self, ids:list[str])->None:
        self.collection.delete(ids=ids)

    def query_by_ids(
        self,
        query_ids: list[str]
    ) -> dict[str, list]:
        results = self.collection.get(
            ids=query_ids,
            include=["documents", "metadatas", "embeddings"]
        )
        return {
            'ids': results['ids'],
            'metadatas': [self.ChromaMetaData(id=metadata['id'],path=metadata['path'],document=metadata['document'],is_success=metadata['is_success']) for metadata in results['metadatas']],
            'documents':[self.ChromaDocument(document=document) for document in results['documents']],
            'embeddings': results['embeddings'],
        }
    
    def query_by_embeddings(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        distance_threshold: float | None = None
    ) -> dict[str, list]:
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances", "embeddings"]
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

    with open('captions_20250522_013210.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    succeed_data = [item for item in data if item['is_success']]

    # ChromaMetaDataインスタンスを作成
    metadatas = [
        ChromaDBManager.ChromaMetaData(
            path=item["path"],
            document=item["caption"],
            is_success=item["is_success"]
        )
        for item in succeed_data
    ]

    # Sentence Embedding（テキストベース）
    sentence_db_manager = ChromaDBManager("sentence_embeddings")

    sentence_db_manager.add(
        documents=[meta.document for meta in metadatas],  # または meta.caption にしたければ ChromaMetaData にフィールド追加が必要
        metadatas=metadatas,
        embeddings=[SentenceEmbeddingsManager.sentence_to_embedding(item["caption"]) for item in succeed_data]
    )
     
    # Image Embedding（画像ベース）
    image_db_manager = ChromaDBManager("image_embeddings")
    image_db_manager.add(
        documents=[meta.path for meta in metadatas],
        embeddings=[ImageEmbeddingsManager.image_to_embedding(f"./imgs/{item['path']}") for item in succeed_data],
        metadatas=metadatas
    )
    