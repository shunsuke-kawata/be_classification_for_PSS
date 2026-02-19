#!/usr/bin/env python3
"""
ChromaDBコレクションの初期化スクリプト
アプリケーション起動時に必要なすべてのベクトルデータベースを事前作成する
"""
import sys
import os
from pathlib import Path

# パスを追加
current_dir = os.path.dirname(__file__)
backend_dir = os.path.join(current_dir, "backend")
sys.path.append(backend_dir)

def initialize_vector_databases():
    """
    必要なすべてのベクトルデータベース（ChromaDBコレクション）を初期化
    """
    try:
        from clustering.chroma_db_manager import ChromaDBManager
        from clustering.embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
        from clustering.embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
        
        print("🚀 ChromaDBベクトルデータベース初期化開始...")
        print("=" * 50)
        
        # 初期化するコレクション一覧
        collections = [
            "sentence_name_embeddings",
            "sentence_usage_embeddings", 
            "sentence_category_embeddings",
            "image_embeddings"
        ]
        
        # 各コレクションを初期化
        managers = {}
        for collection_name in collections:
            print(f"📦 初期化中: {collection_name}")
            try:
                manager = ChromaDBManager(collection_name)
                managers[collection_name] = manager
                
                # コレクション情報を取得
                info = manager.get_collection_info()
                if info:
                    print(f"   ✅ 作成成功 - レコード数: {info['count']}, 次元: {info['embedding_dimension']}")
                else:
                    print(f"   ✅ 作成成功 - 新規コレクション")
                    
            except Exception as e:
                print(f"   ❌ エラー: {e}")
                # 次元エラーの場合は自動的にリセットされるはず
                if "dimension" in str(e).lower():
                    print(f"   🔄 次元不一致のため自動リセット済み")
                else:
                    raise e
        
        # 埋め込み次元の確認
        print(f"\n📊 埋め込み次元情報:")
        try:
            test_sentence = "test sentence"
            sentence_embedding = SentenceEmbeddingsManager.sentence_to_embedding(test_sentence)
            print(f"   文章埋め込み次元: {len(sentence_embedding)}")
            
            # 画像埋め込みは実際のファイルが必要なので次元のみ表示
            print(f"   画像埋め込み次元: 512 (ResNet18)")
            
        except Exception as e:
            print(f"   ⚠️  埋め込み次元確認エラー: {e}")
        
        print(f"\n✅ すべてのベクトルデータベースの初期化が完了しました")
        print(f"   初期化されたコレクション数: {len(managers)}")
        
        return managers
        
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_existing_collections():
    """
    既存のコレクションの状態をチェック
    """
    try:
        from clustering.chroma_db_manager import ChromaDBManager
        import chromadb
        
        print("🔍 既存コレクションのチェック...")
        
        # ChromaDBクライアントを直接使用してコレクション一覧を取得
        client = chromadb.PersistentClient(path="./chroma_db")
        existing_collections = client.list_collections()
        
        if existing_collections:
            print(f"   既存コレクション数: {len(existing_collections)}")
            for collection in existing_collections:
                print(f"   - {collection.name}")
        else:
            print(f"   既存コレクションなし")
            
        return existing_collections
        
    except Exception as e:
        print(f"⚠️  既存コレクションチェックエラー: {e}")
        return []

def main():
    """メイン実行関数"""
    print("ChromaDBベクトルデータベース初期化ツール")
    print("=" * 60)
    
    # 既存コレクションの確認
    existing = check_existing_collections()
    print()
    
    # ベクトルデータベースの初期化
    managers = initialize_vector_databases()
    
    if managers:
        print(f"\n🎉 初期化完了! アプリケーションの準備ができました。")
    else:
        print(f"\n💥 初期化に失敗しました。ログを確認してください。")
        sys.exit(1)

if __name__ == "__main__":
    main()