"""
単語分析モジュール

文節から単語を抽出し、上位語を取得して類似度を計算する機能を提供
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import spacy
from nltk.corpus import wordnet as wn
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, util

from config import CAPTION_STOPWORDS


class WordAnalyzer:
    """単語分析クラス"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        """
        初期化
        
        Args:
            embedding_model: 埋め込みモデル
        """
        self.embedding_model = embedding_model
        self.nlp = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """spacy と WordNet を初期化"""
        print(f"    📚 spacy と WordNet を初期化中...")
        
        # spacy モデルをロード
        try:
            self.nlp = spacy.load('en_core_web_md')
            print(f"    ✅ spaCy モデル (en_core_web_md) 読み込み完了")
        except OSError:
            print(f"    ❌ spaCy モデル (en_core_web_md) が見つかりません")
            print(f"    💡 以下のコマンドで手動インストールしてください: python -m spacy download en_core_web_md")
            raise
        
        # nltk の WordNet データを確認（Dockerビルド時にダウンロード済みのはず）
        try:
            nltk.data.find('corpora/wordnet')
            print(f"    ✅ WordNet データ読み込み完了")
        except LookupError:
            print(f"    📥 WordNet データが見つかりません。ダウンロード中...")
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            print(f"    ✅ WordNet データダウンロード完了")
        
        print(f"    ✅ spacy と WordNet の初期化完了")
    
    def get_common_category(self, word1: str, word2: str) -> Tuple[List[str], float]:
        """
        2つの単語の共通カテゴリ（最も近い共通上位概念）を取得
        
        Args:
            word1: 単語1
            word2: 単語2
            
        Returns:
            (共通カテゴリ名のリスト, スコア)
        """
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)

        if not synsets1 or not synsets2:
            return [], -1
        
        best_pair = None
        best_score = -1

        # 全ての意味の組み合わせを比較
        for s1 in synsets1:
            for s2 in synsets2:
                # 最も近い共通上位概念を取得
                common = s1.lowest_common_hypernyms(s2)
                if not common:
                    continue
                
                # "距離が近いほど一般カテゴリとして適切" とみなす
                # （synset に定義された深さを使う）
                score = max([c.min_depth() for c in common])

                if score > best_score:
                    best_score = score
                    best_pair = common

        if best_pair:
            # 最も代表的なカテゴリ名を返す
            category_names = [c.name().split('.')[0] for c in best_pair]
            return category_names, best_score
        
        return [], -1
    
    @staticmethod
    def extract_words(sentence: str) -> List[str]:
        """
        文から単語を抽出（2文字以上の英単語のみ、ストップワードを除外）
        
        Args:
            sentence: 文節
            
        Returns:
            単語のリスト
        """
        stop_words_lower = [w.lower() for w in CAPTION_STOPWORDS]
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence.lower())
        return [w for w in words if w not in stop_words_lower]
    
    def analyze_folder_words(
        self,
        folder_sentences_by_position: Dict[str, Dict[int, List[str]]],
        target_position: int,
        folder_ids: List[str],
        folder_id_to_name: Dict[str, str]
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[Tuple[str, int]]]]:
        """
        各フォルダの文節位置から単語を抽出し、頻度をカウント
        
        Args:
            folder_sentences_by_position: {folder_id: {position: [sentences]}}
            target_position: 対象の文節位置
            folder_ids: フォルダIDのリスト
            folder_id_to_name: {folder_id: folder_name}
            
        Returns:
            (folder_word_frequencies, folder_top_words)
            - folder_word_frequencies: {folder_id: {word: count}}
            - folder_top_words: {folder_id: [(word, freq), ...]}
        """
        folder_word_frequencies = {}
        
        for folder_id in folder_ids:
            if target_position not in folder_sentences_by_position[folder_id]:
                continue
            
            # その文節位置の全文を取得
            target_sentences = folder_sentences_by_position[folder_id][target_position]
            
            # 全文から単語を抽出してカウント
            word_count = {}
            for sentence in target_sentences:
                words = self.extract_words(sentence)
                for word in words:
                    word_count[word] = word_count.get(word, 0) + 1
            
            folder_word_frequencies[folder_id] = word_count
            
            # 使用頻度の高い順にランク付け（トップ10）
            top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
            folder_name = folder_id_to_name[folder_id]
            print(f"      📁 {folder_name}: {len(word_count)}個のユニーク単語")
            top_words_str = ', '.join([f"{w}({c})" for w, c in top_words[:5]])
            print(f"         頻出単語トップ5: {top_words_str}")
        
        return folder_word_frequencies, None
    
    def get_top_unique_words(
        self,
        folder_word_frequencies: Dict[str, Dict[str, int]],
        folder_ids: List[str],
        folder_id_to_name: Dict[str, str],
        top_n: int = 5
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        各フォルダの頻出単語トップNから共通単語を除外
        
        Args:
            folder_word_frequencies: {folder_id: {word: count}}
            folder_ids: フォルダIDのリスト
            folder_id_to_name: {folder_id: folder_name}
            top_n: 上位N件
            
        Returns:
            {folder_id: [(word, freq), ...]}
        """
        # 各フォルダの頻出単語トップNを取得
        folder_top_words = {}
        for folder_id in folder_ids:
            sorted_words = sorted(
                folder_word_frequencies[folder_id].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            folder_top_words[folder_id] = sorted_words
        
        # トップN同士を比較して共通する単語を特定
        top_words_per_folder = [set([w for w, _ in folder_top_words[fid]]) for fid in folder_ids]
        common_words = set.intersection(*top_words_per_folder)
        
        print(f"       トップ{top_n}内で共通する単語数: {len(common_words)}個")
        if len(common_words) > 0:
            common_words_str = ', '.join(sorted(list(common_words)))
            print(f"       共通単語: {common_words_str}")
        
        # 各フォルダの頻出単語トップNから共通単語を除外
        folder_top_unique_words = {}
        
        for folder_id in folder_ids:
            unique_top_words = [
                (w, freq) for w, freq in folder_top_words[folder_id]
                if w not in common_words
            ]
            
            folder_top_unique_words[folder_id] = unique_top_words
            
            folder_name = folder_id_to_name[folder_id]
            print(f"       📁 {folder_name}: {len(unique_top_words)}個の固有単語（トップ{top_n}から共通単語除外後）")
            if len(unique_top_words) > 0:
                top_display = ', '.join([f"{w}({freq})" for w, freq in unique_top_words])
                print(f"          固有単語: {top_display}")
        
        return folder_top_unique_words
    
    def compute_common_category_similarity(
        self,
        folder_top_unique_words: Dict[str, List[Tuple[str, int]]],
        folder_ids: List[str],
        folder_id_to_name: Dict[str, str],
        similarity_threshold: float = 0.5
    ) -> Tuple[List[dict], List[dict]]:
        """
        共通カテゴリベースの類似度マッチング
        
        Args:
            folder_top_unique_words: {folder_id: [(word, freq), ...]}
            folder_ids: フォルダIDのリスト
            folder_id_to_name: {folder_id: folder_name}
            similarity_threshold: 類似度の閾値
            
        Returns:
            (all_category_pairs, similar_category_pairs)
        """
        print(f"\n    🔍 ステップ6: 共通カテゴリベースの類似度マッチング...")
        
        # 全ての単語ペアを収集
        all_words_with_freq = []
        for folder_id in folder_ids:
            if folder_id not in folder_top_unique_words:
                continue
            for word, freq in folder_top_unique_words[folder_id]:
                all_words_with_freq.append({
                    'folder_id': folder_id,
                    'word': word,
                    'freq': freq
                })
        
        if len(all_words_with_freq) < 2:
            print(f"       ⚠️ 比較する単語が不足しています（{len(all_words_with_freq)}個）")
            return [], []
        
        # フォルダ間で共通カテゴリを計算
        all_category_pairs = []
        skipped_pairs = []  # スキップされたペアを記録
        
        for i, item1 in enumerate(all_words_with_freq):
            for j, item2 in enumerate(all_words_with_freq):
                if i >= j:
                    continue
                
                # 異なるフォルダ間のみ比較
                if item1['folder_id'] == item2['folder_id']:
                    continue
                
                folder1_name = folder_id_to_name[item1['folder_id']]
                folder2_name = folder_id_to_name[item2['folder_id']]
                
                # 共通カテゴリを取得
                common_categories, category_score = self.get_common_category(
                    item1['word'],
                    item2['word']
                )
                
                if len(common_categories) == 0 or category_score < 0:
                    # スキップされたペアを記録
                    skipped_pairs.append({
                        'word1': item1['word'],
                        'word2': item2['word'],
                        'folder1': folder1_name,
                        'folder2': folder2_name,
                        'freq1': item1['freq'],
                        'freq2': item2['freq']
                    })
                    continue
                
                # 単語の埋め込みベクトルで類似度を計算
                word1_embedding = self.embedding_model.encode([item1['word']], convert_to_tensor=True)
                word2_embedding = self.embedding_model.encode([item2['word']], convert_to_tensor=True)
                word_similarity = util.cos_sim(word1_embedding, word2_embedding).item()
                
                pair_info = {
                    'word1': item1['word'],
                    'word2': item2['word'],
                    'folder1_id': item1['folder_id'],
                    'folder2_id': item2['folder_id'],
                    'common_categories': common_categories,
                    'category_score': category_score,
                    'word_similarity': word_similarity,
                    'freq1': item1['freq'],
                    'freq2': item2['freq'],
                    'freq_sum': item1['freq'] + item2['freq']
                }
                
                all_category_pairs.append(pair_info)
                
                category_display = ', '.join(common_categories[:3])
                print(f"         🔄 {folder1_name} '{item1['word']}' ↔ {folder2_name} '{item2['word']}'")
                print(f"            共通カテゴリ: {category_display} (スコア: {category_score:.2f})")
                print(f"            単語類似度: {word_similarity:.3f}, 頻度: {item1['freq']}, {item2['freq']}")
        
        # スキップされたペアをログ出力
        if len(skipped_pairs) > 0:
            print(f"\n       ⚠️ 共通カテゴリが見つからずスキップされたペア: {len(skipped_pairs)}個")
            for skip in skipped_pairs:
                print(f"         ❌ {skip['folder1']} '{skip['word1']}' (頻度:{skip['freq1']}) ↔ {skip['folder2']} '{skip['word2']}' (頻度:{skip['freq2']})")
        
        if len(all_category_pairs) == 0:
            print(f"       ⚠️ 共通カテゴリを持つペアが見つかりませんでした")
            if len(skipped_pairs) > 0:
                print(f"       💡 ヒント: WordNetで共通カテゴリが見つからない単語ペアが多い可能性があります")
                print(f"       💡 解決策: より一般的な単語を使用するか、類似度閾値を下げることを検討してください")
            return [], []
        
        # カテゴリスコアでフィルタリング（スコアが高いほど近いカテゴリ）
        # まず、category_scoreが最低閾値（例: 1以上）のペアのみを対象
        min_category_score = 1
        filtered_pairs = [p for p in all_category_pairs if p['category_score'] >= min_category_score]
        
        print(f"\n       📊 共通カテゴリの統計:")
        print(f"          全ペア数: {len(all_category_pairs)}")
        print(f"          カテゴリスコア >= {min_category_score} のペア数: {len(filtered_pairs)}")
        
        if len(filtered_pairs) == 0:
            print(f"       ⚠️ 有効な共通カテゴリを持つペアが見つかりませんでした")
            return all_category_pairs, []
        
        # ソート優先順位:
        # 1. カテゴリスコア（高いほど良い）
        # 2. 単語ベクトル類似度（高いほど良い）
        # 3. 頻度合計（高いほど良い）
        sorted_pairs = sorted(
            filtered_pairs,
            key=lambda x: (x['category_score'], x['word_similarity'], x['freq_sum']),
            reverse=True
        )
        
        # 類似度閾値も考慮（単語類似度が閾値以上のもの）
        similar_category_pairs = [p for p in sorted_pairs if p['word_similarity'] >= similarity_threshold]
        
        print(f"          単語類似度 >= {similarity_threshold} のペア数: {len(similar_category_pairs)}")
        
        # 類似度閾値を満たすペアがない場合は、上位のペアを返す
        if len(similar_category_pairs) == 0:
            print(f"       ℹ️ 類似度閾値を満たすペアがないため、上位3ペアを採用")
            similar_category_pairs = sorted_pairs[:min(3, len(sorted_pairs))]
        
        return all_category_pairs, similar_category_pairs
    
    def select_representative_words(
        self,
        similar_category_pairs: List[dict],
        folder_ids: List[str],
        folder_id_to_name: Dict[str, str],
        folder_top_unique_words: Dict[str, List[Tuple[str, int]]]
    ) -> Dict[str, str]:
        """
        各フォルダの代表単語を選択
        
        Args:
            similar_category_pairs: 類似する共通カテゴリペアのリスト
            folder_ids: フォルダIDのリスト
            folder_id_to_name: {folder_id: folder_name}
            folder_top_unique_words: {folder_id: [(word, freq), ...]}
            
        Returns:
            {folder_id: representative_word}
        """
        if len(similar_category_pairs) == 0:
            return {}
        
        print(f"\n       🎯 共通カテゴリを持つ単語ペア:")
        for pair in similar_category_pairs:
            folder1_name = folder_id_to_name[pair['folder1_id']]
            folder2_name = folder_id_to_name[pair['folder2_id']]
            category_display = ', '.join(pair['common_categories'][:3])
            print(f"         '{pair['word1']}' ({folder1_name}, 頻度:{pair['freq1']}) ↔ '{pair['word2']}' ({folder2_name}, 頻度:{pair['freq2']})")
            print(f"           共通カテゴリ: {category_display} (スコア: {pair['category_score']:.2f})")
            print(f"           単語類似度: {pair['word_similarity']:.3f}, 頻度合計: {pair['freq_sum']}")
        
        # 各フォルダの代表単語を選択
        # 優先順位: カテゴリスコア > 単語類似度 > 頻度
        folder_word_candidates = defaultdict(list)
        
        for pair in similar_category_pairs:
            folder_word_candidates[pair['folder1_id']].append(
                (pair['word1'], pair['freq1'], pair['category_score'], pair['word_similarity'])
            )
            folder_word_candidates[pair['folder2_id']].append(
                (pair['word2'], pair['freq2'], pair['category_score'], pair['word_similarity'])
            )
        
        folder_representative_words = {}
        
        for folder_id in folder_ids:
            if folder_id in folder_word_candidates and len(folder_word_candidates[folder_id]) > 0:
                # カテゴリスコア > 単語類似度 > 頻度 の順でソート
                sorted_candidates = sorted(
                    folder_word_candidates[folder_id],
                    key=lambda x: (x[2], x[3], x[1]),  # (category_score, word_similarity, freq)
                    reverse=True
                )
                best_word = sorted_candidates[0][0]
                folder_representative_words[folder_id] = best_word
        
        if len(folder_representative_words) > 0:
            classification_words = list(set(folder_representative_words.values()))
            
            print(f"\n       🎯 フォルダ分類基準（共通カテゴリベース）: {classification_words}")
            print(f"       各フォルダの代表単語:")
            for folder_id, word in folder_representative_words.items():
                folder_name = folder_id_to_name[folder_id]
                freq = dict(folder_top_unique_words[folder_id])[word]
                
                # 共通カテゴリを探す
                matching_pairs = [p for p in similar_category_pairs 
                                 if (p['folder1_id'] == folder_id and p['word1'] == word) 
                                 or (p['folder2_id'] == folder_id and p['word2'] == word)]
                
                if matching_pairs:
                    best_match = matching_pairs[0]
                    category_display = ', '.join(best_match['common_categories'][:3])
                    print(f"         📁 {folder_name}: {word} (使用頻度: {freq}回, 共通カテゴリ: {category_display})")
                else:
                    print(f"         📁 {folder_name}: {word} (使用頻度: {freq}回)")
        
        return folder_representative_words
