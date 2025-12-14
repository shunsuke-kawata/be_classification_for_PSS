"""
継続的階層分類のレポート生成クラス

実行ごとにテキストレポートを生成し、指定されたディレクトリ構造に保存する。
ディレクトリ構造: output/{project_name}/{user_name}/{yyyymmddhhmmss}/{image_name}.txt
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from clustering.clustering_metrics import ClusteringMetrics


class ContinuousClusteringReporter:
    """継続的階層分類のレポート生成クラス"""
    
    def __init__(self, project_name: str, user_name: str, output_base_dir: str = "output"):
        """
        Args:
            project_name: プロジェクト名
            user_name: ユーザー名
            output_base_dir: 出力ベースディレクトリ（デフォルト: "output"）
        """
        self.project_name = self._sanitize_dirname(project_name)
        self.user_name = self._sanitize_dirname(user_name)
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # ディレクトリパスを構築
        self.report_dir = Path(output_base_dir) / self.project_name / self.user_name / self.timestamp
        
        # メトリクス計算クラスを初期化
        self.metrics_calculator = ClusteringMetrics()
        
        # ディレクトリを作成
        self._create_directories()
        
    def _sanitize_dirname(self, name: str) -> str:
        """
        ディレクトリ名として使用可能な文字列に変換
        
        Args:
            name: 元の名前
            
        Returns:
            サニタイズされた名前
        """
        # 使用できない文字を置換
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        sanitized = name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized
    
    def _create_directories(self):
        """レポート出力用ディレクトリを作成"""
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 レポートディレクトリを作成しました: {self.report_dir}")
        except Exception as e:
            print(f"❌ ディレクトリ作成エラー: {e}")
            raise
    
    def generate_image_report(self, report_data: Dict[str, Any]) -> str:
        """
        画像ごとのレポートを生成
        
        Args:
            report_data: レポートデータ（辞書形式）
            
        Returns:
            生成されたレポートのファイルパス
        """
        # デバッグ: レポートデータのキーを出力
        print(f"  🔍 レポートデータのキー: {list(report_data.keys())}")
        
        image_name = report_data.get('image_name', 'unknown')
        safe_image_name = self._sanitize_dirname(image_name)
        
        # 拡張子を除去してtxtに置換
        base_name = os.path.splitext(safe_image_name)[0]
        report_filename = f"{base_name}.txt"
        report_path = self.report_dir / report_filename
        
        print(f"  📝 レポート生成開始: {report_filename}")
        
        # レポート内容を生成
        report_content = self._format_report(report_data)
        
        # ファイルに書き込み
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"  📄 レポート作成: {report_path}")
            return str(report_path)
        except Exception as e:
            print(f"  ❌ レポート作成エラー: {e}")
            raise
    
    def _format_report(self, data: Dict[str, Any]) -> str:
        """
        レポートデータをフォーマット
        
        Args:
            data: レポートデータ
            
        Returns:
            フォーマットされたレポート文字列
        """
        lines = []
        lines.append("=" * 80)
        lines.append("継続的階層分類レポート")
        lines.append("=" * 80)
        lines.append("")
        
        # 実行情報
        lines.append("【実行情報】")
        lines.append(f"  実行日時: {data.get('execution_time', 'N/A')}")
        lines.append(f"  プロジェクト: {data.get('project_name', 'N/A')}")
        lines.append(f"  ユーザー: {data.get('user_name', 'N/A')}")
        lines.append(f"  クラスタリング回数: {data.get('clustering_count', 'N/A')}")
        lines.append("")
        
        # 画像情報
        lines.append("【画像情報】")
        lines.append(f"  画像ID: {data.get('image_id', 'N/A')}")
        lines.append(f"  ファイル名: {data.get('image_name', 'N/A')}")
        lines.append(f"  Clustering ID: {data.get('clustering_id', 'N/A')}")
        lines.append(f"  ChromaDB Sentence ID: {data.get('chromadb_sentence_id', 'N/A')}")
        lines.append(f"  ChromaDB Image ID: {data.get('chromadb_image_id', 'N/A')}")
        lines.append("")
        
        # キャプション情報
        caption = data.get('caption', 'N/A')
        lines.append("【キャプション】")
        lines.append(f"  {caption}")
        lines.append("")
        
        # 埋め込みベクトル情報
        lines.append("【埋め込みベクトル情報】")
        lines.append(f"  文章ベクトル取得: {'成功' if data.get('sentence_embedding_available', False) else '失敗'}")
        lines.append(f"  画像ベクトル取得: {'成功' if data.get('image_embedding_available', False) else '失敗'}")
        lines.append("")
        
        # 類似度計算結果
        lines.append("【類似度計算結果】")
        lines.append(f"  対象フォルダ数: {data.get('total_folders_checked', 0)}")
        lines.append("")
        
        # 上位の類似度スコア
        similarity_scores = data.get('similarity_scores', [])
        if similarity_scores:
            lines.append("  上位の類似度スコア:")
            for i, score_info in enumerate(similarity_scores[:10], 1):  # 上位10件
                folder_name = score_info.get('folder_name', 'Unknown')
                folder_id = score_info.get('folder_id', 'N/A')
                similarity = score_info.get('similarity', 0.0)
                sim_type = score_info.get('type', 'N/A')
                lines.append(f"    [{i}] {folder_name} (ID: {folder_id})")
                lines.append(f"        類似度: {similarity:.6f} (タイプ: {sim_type})")
            lines.append("")
        
        # 最終的な分類先
        lines.append("【分類結果】")
        final_folder_name = data.get('final_folder_name', 'N/A')
        final_folder_id = data.get('final_folder_id', 'N/A')
        final_similarity = data.get('final_similarity', 0.0)
        final_similarity_type = data.get('final_similarity_type', 'N/A')
        
        lines.append(f"  分類先フォルダ: {final_folder_name}")
        lines.append(f"  フォルダID: {final_folder_id}")
        lines.append(f"  最終類似度: {final_similarity:.6f}")
        lines.append(f"  類似度タイプ: {final_similarity_type}")
        lines.append("")
        
        # フォルダ平均との類似度（既存フォルダに追加された場合）
        if 'folder_average_sentence_similarity' in data or 'folder_average_image_similarity' in data:
            lines.append("【フォルダ平均との類似度】")
            lines.append("  ※既存フォルダに追加されたため、フォルダ内画像の平均ベクトルとの類似度")
            if 'folder_average_sentence_similarity' in data:
                sent_sim = data['folder_average_sentence_similarity']
                lines.append(f"  文章特徴量の類似度: {sent_sim:.6f}")
            if 'folder_average_image_similarity' in data:
                img_sim = data['folder_average_image_similarity']
                lines.append(f"  画像特徴量の類似度: {img_sim:.6f}")
            lines.append("")
        
        # 新規フォルダ作成情報
        if data.get('new_folder_created', False):
            lines.append("【新規フォルダ作成】")
            lines.append(f"  ✓ 新しいフォルダを作成しました")
            lines.append(f"  フォルダ名: {data.get('new_folder_name', 'N/A')}")
            lines.append(f"  フォルダID: {data.get('new_folder_id', 'N/A')}")
            lines.append(f"  作成理由: 類似度が閾値 {data.get('similarity_threshold', 0.4)} を下回ったため")
            lines.append("")
        
        # 分類基準による再判定情報
        criteria_used = data.get('classification_criteria_used', False)
        if criteria_used:
            lines.append("【分類基準による再判定】")
            lines.append(f"  再判定実行: はい")
            
            classification_words = data.get('classification_words_found', [])
            if classification_words:
                lines.append(f"  検出された分類キーワード:")
                for word_info in classification_words:
                    word = word_info.get('word', 'N/A')
                    count = word_info.get('count', 0)
                    target_folder = word_info.get('target_folder', 'N/A')
                    lines.append(f"    - '{word}' (出現回数: {count}, 対象フォルダ: {target_folder})")
            
            override_folder = data.get('criteria_target_folder_name', None)
            if override_folder:
                lines.append(f"  再判定結果フォルダ: {override_folder}")
            lines.append("")
        
        # 兄弟フォルダのTF-IDFスコア表（分類基準処理が実行された場合）
        if data.get('classification_criteria_process_executed', False):
            sibling_tfidf = data.get('sibling_folder_tfidf_scores', {})
            if sibling_tfidf:
                lines.append("【兄弟フォルダの特徴的単語（TF-IDFスコア表）】")
                lines.append("  ※各フォルダを最も代表する単語とそのスコア")
                lines.append("")
                
                # フォルダごとにスコア表を出力
                for folder_id, folder_info in sibling_tfidf.items():
                    folder_name = folder_info.get('folder_name', 'Unknown')
                    unique_words = folder_info.get('unique_words', [])
                    
                    lines.append(f"  📁 {folder_name} (ID: {folder_id})")
                    lines.append(f"     順位 | 単語              | 総合  | 代表性 | 識別性 | TF     | 集中度 | 一貫性 | IDF  ")
                    lines.append(f"     " + "-" * 95)
                    
                    for idx, word_data in enumerate(unique_words[:10], 1):  # 上位10個
                        word = word_data.get('word', '')
                        score = float(word_data.get('score', 0.0))
                        score_repr = float(word_data.get('score_repr', 0.0))
                        score_dist = float(word_data.get('score_dist', 0.0))
                        tf = float(word_data.get('tf', 0.0))
                        concentration = float(word_data.get('concentration', 0.0))
                        consistency = float(word_data.get('consistency', 0.0))
                        base_idf = float(word_data.get('base_idf', 0.0))
                        
                        lines.append(f"     {idx:2d}   | {word:16s} | {score:5.1f} | {score_repr:6.1f} | {score_dist:6.1f} | {tf:6.4f} | {concentration:6.4f} | {consistency:6.4f} | {base_idf:4.2f}")
                    
                    lines.append("")
                    lines.append(f"     ※ 総合スコア = 0.7 × 代表性 + 0.3 × 識別性")
                    lines.append(f"     ※ 代表性 = TF × 集中度 × √一貫性 × 1000: フォルダの内容を表す単語")
                    lines.append(f"     ※ 識別性 = TF × IDF × 集中度 × 100: 他フォルダと区別する単語")
                    lines.append(f"     ※ TF = 文位置重み付き出現比率, 集中度 = このフォルダ出現/全体出現, 一貫性 = 単語を含む画像率")
                    lines.append(f"     ※ IDF = log((総フォルダ数+1)/(出現フォルダ数+1)): グローバルな希少性")
                    lines.append("")
                
                # 分類基準の詳細情報も追加
                criteria_details = data.get('classification_criteria_details', {})
                if criteria_details:
                    lines.append("  【分類基準の詳細】")
                    for category, info in criteria_details.items():
                        rank = info.get('rank', '-')
                        avg_score = float(info.get('avg_score', 0.0))
                        word_count = int(info.get('word_count', 0))
                        words = info.get('words', [])
                        lines.append(f"    第{rank}位: {category}")
                        lines.append(f"      平均スコア: {avg_score:.2f}, 単語数: {word_count}")
                        lines.append(f"      単語: {', '.join(words[:10])}")
                    lines.append("")
        
        # 詳細な特徴分析結果
        feature_analysis = data.get('feature_analysis', {})
        if feature_analysis:
            lines.append("【特徴分析結果】")
            
            # TF-IDFスコア
            tfidf_scores = feature_analysis.get('tfidf_scores', {})
            if tfidf_scores:
                lines.append("  TF-IDFスコア:")
                for folder_name, scores in tfidf_scores.items():
                    lines.append(f"    {folder_name}:")
                    for word, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                        lines.append(f"      - {word}: {score:.6f}")
            
            # 色情報
            color_info = feature_analysis.get('color_analysis', {})
            if color_info:
                lines.append("  色分析:")
                for key, value in color_info.items():
                    lines.append(f"    {key}: {value}")
            
            # 形状情報
            shape_info = feature_analysis.get('shape_analysis', {})
            if shape_info:
                lines.append("  形状分析:")
                for key, value in shape_info.items():
                    lines.append(f"    {key}: {value}")
            
            lines.append("")
        
        # 同階層フォルダ情報
        sibling_info = data.get('sibling_folders_info', {})
        if sibling_info:
            lines.append("【同階層フォルダ情報】")
            lines.append(f"  同階層フォルダ数: {sibling_info.get('total_siblings', 0)}")
            lines.append(f"  リーフフォルダ数: {sibling_info.get('leaf_siblings', 0)}")
            
            sibling_list = sibling_info.get('sibling_list', [])
            if sibling_list:
                lines.append("  フォルダ一覧:")
                for sib in sibling_list:
                    lines.append(f"    - {sib.get('name', 'N/A')} (ID: {sib.get('id', 'N/A')}, Leaf: {sib.get('is_leaf', False)})")
            lines.append("")
        
        # その他の情報
        additional_info = data.get('additional_info', {})
        if additional_info:
            lines.append("【追加情報】")
            for key, value in additional_info.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # エラー情報
        errors = data.get('errors', [])
        if errors:
            lines.append("【エラー・警告】")
            for error in errors:
                lines.append(f"  ⚠ {error}")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("レポート終了")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_summary_report(self, all_reports_data: List[Dict[str, Any]]) -> str:
        """
        実行全体のサマリーレポートを生成
        
        Args:
            all_reports_data: 全画像のレポートデータのリスト
            
        Returns:
            サマリーレポートのファイルパス
        """
        summary_path = self.report_dir / "SUMMARY.txt"
        
        lines = []
        lines.append("=" * 80)
        lines.append("継続的階層分類 実行サマリー")
        lines.append("=" * 80)
        lines.append("")
        
        # 実行情報
        if all_reports_data:
            first_report = all_reports_data[0]
            lines.append("【実行情報】")
            lines.append(f"  実行日時: {first_report.get('execution_time', 'N/A')}")
            lines.append(f"  プロジェクト: {first_report.get('project_name', 'N/A')}")
            lines.append(f"  ユーザー: {first_report.get('user_name', 'N/A')}")
            lines.append(f"  クラスタリング回数: {first_report.get('clustering_count', 'N/A')}")
            lines.append("")
        
        # 統計情報
        lines.append("【統計情報】")
        lines.append(f"  処理画像数: {len(all_reports_data)}")
        
        new_folders_count = sum(1 for r in all_reports_data if r.get('new_folder_created', False))
        lines.append(f"  新規作成フォルダ数: {new_folders_count}")
        
        criteria_used_count = sum(1 for r in all_reports_data if r.get('classification_criteria_used', False))
        lines.append(f"  分類基準による再判定: {criteria_used_count}件")
        
        # 平均類似度
        similarities = [r.get('final_similarity', 0.0) for r in all_reports_data if r.get('final_similarity')]
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            min_similarity = min(similarities)
            lines.append(f"  平均類似度: {avg_similarity:.6f}")
            lines.append(f"  最高類似度: {max_similarity:.6f}")
            lines.append(f"  最低類似度: {min_similarity:.6f}")
        lines.append("")
        
        # 画像ごとの簡易サマリー
        lines.append("【処理画像一覧】")
        for i, report in enumerate(all_reports_data, 1):
            image_name = report.get('image_name', 'Unknown')
            folder_name = report.get('final_folder_name', 'N/A')
            similarity = report.get('final_similarity', 0.0)
            new_folder = "✓" if report.get('new_folder_created', False) else ""
            lines.append(f"  [{i}] {image_name}")
            lines.append(f"      → {folder_name} (類似度: {similarity:.4f}) {new_folder}")
        
        lines.append("")
        lines.append("=" * 80)
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            print(f"📄 サマリーレポート作成: {summary_path}")
            return str(summary_path)
        except Exception as e:
            print(f"❌ サマリーレポート作成エラー: {e}")
            raise
    
    def generate_metrics_report(
        self,
        all_reports_data: List[Dict[str, Any]],
        folder_data: Dict[str, Any] = None,
        similarity_threshold: float = 0.4
    ) -> str:
        """
        評価指標レポートを生成
        
        Args:
            all_reports_data: 全画像のレポートデータのリスト
            folder_data: フォルダ構造データ
            similarity_threshold: 類似度閾値
            
        Returns:
            評価指標レポートのファイルパス
        """
        metrics_path = self.report_dir / "METRICS_REPORT.txt"
        
        # 評価指標を計算
        metrics = self.metrics_calculator.calculate_all_metrics(
            all_reports_data,
            folder_data or {},
            similarity_threshold
        )
        
        lines = []
        lines.append("=" * 80)
        lines.append("継続的階層分類 評価指標レポート")
        lines.append("=" * 80)
        lines.append("")
        
        # 実行情報
        if all_reports_data:
            first_report = all_reports_data[0]
            lines.append("【実行情報】")
            lines.append(f"  実行日時: {first_report.get('execution_time', 'N/A')}")
            lines.append(f"  プロジェクト: {first_report.get('project_name', 'N/A')}")
            lines.append(f"  ユーザー: {first_report.get('user_name', 'N/A')}")
            lines.append(f"  クラスタリング回数: {first_report.get('clustering_count', 'N/A')}")
            lines.append(f"  類似度閾値: {similarity_threshold}")
            lines.append("")
        
        # 1. 基本統計
        if 'basic_stats' in metrics:
            lines.append("=" * 80)
            lines.append("1. 基本統計")
            lines.append("=" * 80)
            stats = metrics['basic_stats']
            lines.append(f"  処理画像総数: {stats.get('total_images', 0)}")
            lines.append(f"  新規フォルダ作成数: {stats.get('new_folders_created', 0)}")
            lines.append(f"  新規フォルダ作成率: {stats.get('new_folder_ratio', 0):.2%}")
            lines.append(f"  既存フォルダへの分類数: {stats.get('existing_folder_assignments', 0)}")
            lines.append(f"  既存フォルダへの分類率: {stats.get('existing_folder_ratio', 0):.2%}")
            lines.append(f"  分類基準使用回数: {stats.get('criteria_based_classifications', 0)}")
            lines.append(f"  分類基準使用率: {stats.get('criteria_usage_ratio', 0):.2%}")
            lines.append(f"  エラー発生回数: {stats.get('errors_occurred', 0)}")
            lines.append(f"  エラー率: {stats.get('error_ratio', 0):.2%}")
            lines.append("")
        
        # 2. 分類品質指標
        if 'classification_success' in metrics:
            lines.append("=" * 80)
            lines.append("2. 分類品質指標")
            lines.append("=" * 80)
            success = metrics['classification_success']
            lines.append(f"  適切な分類数（閾値基準）: {success.get('appropriate_classifications', 0)}")
            lines.append(f"  適切な分類率: {success.get('appropriate_classification_ratio', 0):.2%}")
            lines.append(f"  高信頼度での既存フォルダ分類: {success.get('high_confidence_existing_folder', 0)}")
            lines.append(f"  高信頼度既存フォルダ分類率: {success.get('high_confidence_existing_ratio', 0):.2%}")
            lines.append(f"  適切な新規フォルダ作成: {success.get('appropriate_new_folders', 0)}")
            lines.append(f"  適切な新規フォルダ作成率: {success.get('appropriate_new_folder_ratio', 0):.2%}")
            lines.append("")
        
        # 3. 類似度統計
        if 'similarity_stats' in metrics:
            lines.append("=" * 80)
            lines.append("3. 類似度統計")
            lines.append("=" * 80)
            sim_stats = metrics['similarity_stats']
            lines.append(f"  平均類似度: {sim_stats.get('mean_similarity', 0):.6f}")
            lines.append(f"  中央値類似度: {sim_stats.get('median_similarity', 0):.6f}")
            lines.append(f"  標準偏差: {sim_stats.get('std_similarity', 0):.6f}")
            lines.append(f"  最小類似度: {sim_stats.get('min_similarity', 0):.6f}")
            lines.append(f"  最大類似度: {sim_stats.get('max_similarity', 0):.6f}")
            
            quartiles = sim_stats.get('quartiles', {})
            lines.append(f"  第1四分位数 (Q1): {quartiles.get('q1', 0):.6f}")
            lines.append(f"  第2四分位数 (Q2/中央値): {quartiles.get('q2', 0):.6f}")
            lines.append(f"  第3四分位数 (Q3): {quartiles.get('q3', 0):.6f}")
            
            lines.append(f"  文章ベース分類率: {sim_stats.get('sentence_based_ratio', 0):.2%}")
            lines.append(f"  画像ベース分類率: {sim_stats.get('image_based_ratio', 0):.2%}")
            lines.append("")
        
        # 4. フォルダバランス指標
        if 'folder_balance' in metrics:
            lines.append("=" * 80)
            lines.append("4. フォルダバランス指標")
            lines.append("=" * 80)
            balance = metrics['folder_balance']
            lines.append(f"  使用フォルダ総数: {balance.get('total_folders_used', 0)}")
            lines.append(f"  フォルダあたり平均画像数: {balance.get('mean_images_per_folder', 0):.2f}")
            lines.append(f"  標準偏差: {balance.get('std_images_per_folder', 0):.2f}")
            lines.append(f"  最小画像数: {balance.get('min_images_per_folder', 0)}")
            lines.append(f"  最大画像数: {balance.get('max_images_per_folder', 0)}")
            lines.append(f"  ジニ係数（不均衡度）: {balance.get('gini_coefficient', 0):.4f}")
            lines.append(f"  変動係数 (CV): {balance.get('coefficient_of_variation', 0):.4f}")
            lines.append(f"  バランススコア（0-1）: {balance.get('balance_score', 0):.4f}")
            lines.append("    ※ バランススコアが1に近いほど均等に分散")
            lines.append("")
        
        # 5. 分類基準の一貫性
        if 'criteria_consistency' in metrics:
            lines.append("=" * 80)
            lines.append("5. 分類基準の一貫性")
            lines.append("=" * 80)
            consistency = metrics['criteria_consistency']
            lines.append(f"  分類基準使用回数: {consistency.get('criteria_used_count', 0)}")
            lines.append(f"  成功回数: {consistency.get('criteria_success_count', 0)}")
            lines.append(f"  一貫性スコア: {consistency.get('consistency_score', 0):.4f}")
            lines.append(f"  一貫性パーセンテージ: {consistency.get('consistency_percentage', 0):.2f}%")
            if 'note' in consistency:
                lines.append(f"  備考: {consistency['note']}")
            lines.append("")
        
        # 6. 信頼度スコア
        if 'confidence_scores' in metrics:
            lines.append("=" * 80)
            lines.append("6. 分類信頼度の分布")
            lines.append("=" * 80)
            confidence = metrics['confidence_scores']
            thresholds = confidence.get('thresholds', {})
            lines.append(f"  高信頼度（≥{thresholds.get('high', 0.7)}）:")
            lines.append(f"    件数: {confidence.get('high_confidence_count', 0)}")
            lines.append(f"    割合: {confidence.get('high_confidence_ratio', 0):.2%}")
            lines.append(f"  中信頼度（{thresholds.get('medium', 0.5)} - {thresholds.get('high', 0.7)}）:")
            lines.append(f"    件数: {confidence.get('medium_confidence_count', 0)}")
            lines.append(f"    割合: {confidence.get('medium_confidence_ratio', 0):.2%}")
            lines.append(f"  低信頼度（<{thresholds.get('medium', 0.5)}）:")
            lines.append(f"    件数: {confidence.get('low_confidence_count', 0)}")
            lines.append(f"    割合: {confidence.get('low_confidence_ratio', 0):.2%}")
            lines.append("")
        
        # 7. パフォーマンス指標
        if 'performance' in metrics:
            lines.append("=" * 80)
            lines.append("7. パフォーマンス指標")
            lines.append("=" * 80)
            perf = metrics['performance']
            lines.append(f"  文章埋め込み取得成功率: {perf.get('sentence_embedding_success_rate', 0):.2%}")
            lines.append(f"  画像埋め込み取得成功率: {perf.get('image_embedding_success_rate', 0):.2%}")
            lines.append(f"  両方取得成功数: {perf.get('both_embeddings_available', 0)}")
            lines.append(f"  両方取得成功率: {perf.get('both_embeddings_available_rate', 0):.2%}")
            lines.append("")
        
        # 8. エラー分析
        if 'error_analysis' in metrics:
            lines.append("=" * 80)
            lines.append("8. エラー分析")
            lines.append("=" * 80)
            error_analysis = metrics['error_analysis']
            lines.append(f"  総エラー数: {error_analysis.get('total_errors', 0)}")
            lines.append(f"  エラー発生画像数: {error_analysis.get('images_with_errors', 0)}")
            lines.append(f"  エラーなし率: {error_analysis.get('error_free_ratio', 0):.2%}")
            
            error_types = error_analysis.get('error_type_distribution', {})
            if error_types:
                lines.append("  エラータイプ別分布:")
                for error_type, count in error_types.items():
                    lines.append(f"    - {error_type}: {count}")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("評価指標レポート終了")
        lines.append("=" * 80)
        lines.append("")
        lines.append("【指標の解釈】")
        lines.append("- 新規フォルダ作成率が高い → 既存フォルダとの類似度が低い新しいパターンが多い")
        lines.append("- ジニ係数が低い（バランススコアが高い） → フォルダ間で画像が均等に分散")
        lines.append("- 平均類似度が高い → 既存フォルダとの適合度が高い")
        lines.append("- 変動係数(CV)が低い → フォルダあたりの画像数が安定している")
        lines.append("- 信頼度スコアが高い → 分類の確実性が高い")
        
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            print(f"📊 評価指標レポート作成: {metrics_path}")
            
            # JSON形式でも保存
            json_path = self.report_dir / "METRICS_DATA.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"📊 評価指標データ（JSON）保存: {json_path}")
            
            return str(metrics_path)
        except Exception as e:
            print(f"❌ 評価指標レポート作成エラー: {e}")
            raise

