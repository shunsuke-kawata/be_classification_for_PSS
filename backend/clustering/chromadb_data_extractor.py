"""
ChromaDBからデータを抽出するためのユーティリティスクリプト

使用例:
    python chromadb_data_extractor.py --collection sentence_name_embeddings --limit 10
    python chromadb_data_extractor.py --collection image_embeddings --limit 5 --output vectors.json
"""

import sys
import json
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from clustering.chroma_db_manager import ChromaDBManager


class ChromaDBDataExtractor:
    """ChromaDBからデータを抽出するクラス"""
    
    def __init__(self, collection_name: str):
        """
        Args:
            collection_name: ChromaDBのコレクション名
        """
        self.collection_name = collection_name
        self.db_manager = ChromaDBManager(collection_name)
        
    def extract_vectors(self, limit: int = 10) -> Dict[str, Any]:
        """
        指定された数のベクトルデータを取得
        
        Args:
            limit: 取得するデータの最大数
            
        Returns:
            dict: {
                'collection_name': str,
                'count': int,
                'data': [
                    {
                        'id': str,
                        'embedding': list,
                        'metadata': dict,
                        'document': str
                    }
                ]
            }
        """
        print(f"📊 コレクション '{self.collection_name}' からデータを取得中...")
        
        # コレクション全体を取得
        try:
            collection = self.db_manager.collection
            
            # limitを指定してデータを取得
            results = collection.get(
                limit=limit,
                include=['embeddings', 'metadatas', 'documents']
            )
            
            # データを整形
            extracted_data = {
                'collection_name': self.collection_name,
                'count': len(results['ids']),
                'data': []
            }
            
            for i in range(len(results['ids'])):
                item = {
                    'id': results['ids'][i],
                    'embedding': results['embeddings'][i] if results['embeddings'] else None,
                    'metadata': results['metadatas'][i] if results['metadatas'] else {},
                    'document': results['documents'][i] if results['documents'] else None
                }
                extracted_data['data'].append(item)
            
            print(f"✅ {extracted_data['count']} 件のデータを取得しました")
            return extracted_data
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()
            return {
                'collection_name': self.collection_name,
                'count': 0,
                'data': [],
                'error': str(e)
            }
    
    def print_vectors(self, limit: int = 10, show_full_vector: bool = False):
        """
        ベクトルデータをコンソールに出力
        
        Args:
            limit: 取得するデータの最大数
            show_full_vector: True の場合、ベクトル全体を表示。False の場合、最初の5次元のみ表示
        """
        data = self.extract_vectors(limit)
        
        print("\n" + "="*80)
        print(f"コレクション名: {data['collection_name']}")
        print(f"取得件数: {data['count']}")
        print("="*80 + "\n")
        
        for idx, item in enumerate(data['data'], 1):
            print(f"[{idx}] ID: {item['id']}")
            print(f"    メタデータ: {item['metadata']}")
            print(f"    ドキュメント: {item['document']}")
            
            if item['embedding']:
                vector = item['embedding']
                vector_dim = len(vector)
                
                if show_full_vector:
                    print(f"    ベクトル (次元数: {vector_dim}):")
                    print(f"    {vector}")
                else:
                    print(f"    ベクトル (次元数: {vector_dim}, 最初の5次元):")
                    print(f"    {vector[:5]} ...")
            else:
                print(f"    ベクトル: なし")
            
            print("-" * 80)
    
    def save_to_json(self, limit: int = 10, output_file: str = "chromadb_vectors.json"):
        """
        ベクトルデータをJSONファイルに保存
        
        Args:
            limit: 取得するデータの最大数
            output_file: 出力ファイル名
        """
        data = self.extract_vectors(limit)
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 データを保存しました: {output_path.absolute()}")
        print(f"   ファイルサイズ: {output_path.stat().st_size / 1024:.2f} KB")


def list_collections():
    """利用可能なコレクション一覧を表示"""
    print("\n📋 利用可能なコレクション:")
    collections = [
        "sentence_name_embeddings",
        "sentence_usage_embeddings", 
        "sentence_category_embeddings",
        "image_embeddings"
    ]
    
    for i, col in enumerate(collections, 1):
        print(f"  {i}. {col}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='ChromaDBからベクトルデータを抽出',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # sentence_name_embeddingsから10件取得
  python chromadb_data_extractor.py --collection sentence_name_embeddings --limit 10
  
  # image_embeddingsから5件取得してJSONに保存
  python chromadb_data_extractor.py --collection image_embeddings --limit 5 --output vectors.json
  
  # ベクトル全体を表示
  python chromadb_data_extractor.py --collection sentence_name_embeddings --limit 3 --full
  
  # コレクション一覧を表示
  python chromadb_data_extractor.py --list
        '''
    )
    
    parser.add_argument(
        '--collection', '-c',
        type=str,
        help='コレクション名 (sentence_name_embeddings, image_embeddings など)'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help='取得するデータの最大数 (デフォルト: 10)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='JSONファイルに保存する場合の出力ファイル名'
    )
    
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='ベクトル全体を表示 (デフォルトは最初の5次元のみ)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='利用可能なコレクション一覧を表示'
    )
    
    args = parser.parse_args()
    
    # コレクション一覧表示
    if args.list:
        list_collections()
        return
    
    # コレクション名が指定されていない場合
    if not args.collection:
        print("❌ エラー: --collection オプションでコレクション名を指定してください")
        print()
        list_collections()
        parser.print_help()
        return
    
    # データ抽出
    extractor = ChromaDBDataExtractor(args.collection)
    
    # JSONファイルに保存する場合
    if args.output:
        extractor.save_to_json(limit=args.limit, output_file=args.output)
    
    # コンソールに出力
    extractor.print_vectors(limit=args.limit, show_full_vector=args.full)


if __name__ == "__main__":
    main()
