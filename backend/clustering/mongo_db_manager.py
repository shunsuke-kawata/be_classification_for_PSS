from pymongo import MongoClient
from config import MONGO_AUTH_DB, MONGO_DB, MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASSWORD,MONGO_INITDB_ROOT_PASSWORD,MONGO_INITDB_ROOT_USERNAME

CONNECT_STRING = f"mongodb://{MONGO_INITDB_ROOT_USERNAME}:{MONGO_INITDB_ROOT_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource={MONGO_AUTH_DB}"

class MongoDBManager:
    def __init__(self, db_name: str = MONGO_DB, uri: str = CONNECT_STRING):
        self.client = MongoClient(uri)
        self._db = self.client[db_name]
    
    @property
    def db(self):
        return self._db

    def insert_document(self, collection_name: str, document: dict):
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return result.inserted_id

    def find_documents(self, collection_name: str, query: dict):
        collection = self.db[collection_name]
        return list(collection.find(query))

    def find_one_document(self, collection_name: str, query: dict):
        collection = self.db[collection_name]
        return collection.find_one(query)

    def update_document(self, collection_name: str, query: dict, update: dict, upsert: bool = True):
        collection = self.db[collection_name]
        result = collection.update_one(query, {'$set': update}, upsert=upsert)
        return result.modified_count, result.upserted_id

    def delete_document(self, collection_name: str, query: dict):
        collection = self.db[collection_name]
        result = collection.delete_one(query)
        return result.deleted_count

    def find_one_with_projection(self, collection_name: str, query: dict, projection: dict):
        """プロジェクション（特定のフィールドのみ取得）付きでドキュメントを検索"""
        collection = self.db[collection_name]
        return collection.find_one(query, projection)

    def get_collection(self, collection_name: str):
        """コレクションオブジェクトを直接取得"""
        return self.db[collection_name]

    def unset_field(self, collection_name: str, query: dict, field_path: str):
        """指定されたフィールドを削除する"""
        collection = self.db[collection_name]
        result = collection.update_one(query, {"$unset": {field_path: ""}})
        return result.modified_count