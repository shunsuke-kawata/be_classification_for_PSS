"""
継続的階層分類の評価指標を計算するモジュール

実装する指標:
1. 追加した画像のうち適切な既存フォルダに分類された画像の数・割合
2. 新規フォルダが作成された回数・割合
3. 分類の信頼度（類似度スコアの統計）
4. クラスタリング品質指標（シルエット係数、凝集度など）
5. フォルダ間のバランス（画像数の分散、ジニ係数など）
6. 分類基準の一貫性
7. 処理時間
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict


class ClusteringMetrics:
    """継続的階層分類の評価指標を計算するクラス"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(
        self,
        reports_data: List[Dict[str, Any]],
        folder_data: Dict[str, Any],
        similarity_threshold: float = 0.4
    ) -> Dict[str, Any]:
        """
        全ての評価指標を計算
        
        Args:
            reports_data: 各画像のレポートデータリスト
            folder_data: フォルダ構造データ
            similarity_threshold: 類似度閾値
            
        Returns:
            評価指標の辞書
        """
        if not reports_data:
            return {}
        
        metrics = {}
        
        # 基本統計
        metrics['basic_stats'] = self._calculate_basic_stats(reports_data)
        
        # 分類成功率・新規フォルダ作成率
        metrics['classification_success'] = self._calculate_classification_success(
            reports_data, similarity_threshold
        )
        
        # 類似度統計
        metrics['similarity_stats'] = self._calculate_similarity_stats(reports_data)
        
        # フォルダバランス
        metrics['folder_balance'] = self._calculate_folder_balance(reports_data, folder_data)
        
        # 分類基準の一貫性
        metrics['criteria_consistency'] = self._calculate_criteria_consistency(reports_data)
        
        # 信頼度スコア
        metrics['confidence_scores'] = self._calculate_confidence_scores(reports_data)
        
        # パフォーマンス評価
        metrics['performance'] = self._calculate_performance_metrics(reports_data)
        
        # エラー分析
        metrics['error_analysis'] = self._calculate_error_analysis(reports_data)
        
        # 階層構造の品質
        if folder_data:
            metrics['hierarchy_quality'] = self._calculate_hierarchy_quality(folder_data)
        
        return metrics
    
    def _calculate_basic_stats(self, reports_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基本統計を計算"""
        total_images = len(reports_data)
        
        # 新規フォルダ作成数
        new_folders_count = sum(1 for r in reports_data if r.get('new_folder_created', False))
        
        # 既存フォルダへの分類数
        existing_folders_count = total_images - new_folders_count
        
        # 分類基準使用数
        criteria_used_count = sum(1 for r in reports_data if r.get('classification_criteria_used', False))
        
        # エラー発生数
        error_count = sum(1 for r in reports_data if len(r.get('errors', [])) > 0)
        
        return {
            'total_images': total_images,
            'new_folders_created': new_folders_count,
            'new_folder_ratio': new_folders_count / total_images if total_images > 0 else 0,
            'existing_folder_assignments': existing_folders_count,
            'existing_folder_ratio': existing_folders_count / total_images if total_images > 0 else 0,
            'criteria_based_classifications': criteria_used_count,
            'criteria_usage_ratio': criteria_used_count / total_images if total_images > 0 else 0,
            'errors_occurred': error_count,
            'error_ratio': error_count / total_images if total_images > 0 else 0
        }
    
    def _calculate_classification_success(
        self,
        reports_data: List[Dict[str, Any]],
        threshold: float
    ) -> Dict[str, Any]:
        """
        分類成功率を計算
        
        画像の指標に対応:
        - 適切なフォルダに分類された画像数・割合
        - 新規フォルダ作成数・割合
        """
        total = len(reports_data)
        
        # 閾値以上の類似度で既存フォルダに分類された画像
        high_confidence_existing = sum(
            1 for r in reports_data
            if not r.get('new_folder_created', False) 
            and r.get('final_similarity', 0) >= threshold
        )
        
        # 閾値以下で新規フォルダが作成された画像
        appropriate_new_folders = sum(
            1 for r in reports_data
            if r.get('new_folder_created', False)
            and r.get('final_similarity', 0) < threshold
        )
        
        # 適切に分類された画像の総数
        appropriate_classifications = high_confidence_existing + appropriate_new_folders
        
        return {
            'appropriate_classifications': appropriate_classifications,
            'appropriate_classification_ratio': appropriate_classifications / total if total > 0 else 0,
            'high_confidence_existing_folder': high_confidence_existing,
            'high_confidence_existing_ratio': high_confidence_existing / total if total > 0 else 0,
            'appropriate_new_folders': appropriate_new_folders,
            'appropriate_new_folder_ratio': appropriate_new_folders / total if total > 0 else 0,
            'threshold_used': threshold
        }
    
    def _calculate_similarity_stats(self, reports_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """類似度の統計情報を計算"""
        similarities = [r.get('final_similarity', 0) for r in reports_data]
        
        if not similarities:
            return {}
        
        similarities = np.array(similarities)
        
        # 類似度タイプの分布
        similarity_types = [r.get('final_similarity_type', 'unknown') for r in reports_data]
        type_counter = Counter(similarity_types)
        
        return {
            'mean_similarity': float(np.mean(similarities)),
            'median_similarity': float(np.median(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'quartiles': {
                'q1': float(np.percentile(similarities, 25)),
                'q2': float(np.percentile(similarities, 50)),
                'q3': float(np.percentile(similarities, 75))
            },
            'similarity_type_distribution': dict(type_counter),
            'sentence_based_ratio': type_counter.get('sentence', 0) / len(similarities) if len(similarities) > 0 else 0,
            'image_based_ratio': type_counter.get('image', 0) / len(similarities) if len(similarities) > 0 else 0
        }
    
    def _calculate_folder_balance(
        self,
        reports_data: List[Dict[str, Any]],
        folder_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """フォルダ間のバランスを評価"""
        # 各フォルダに追加された画像数
        folder_assignments = defaultdict(int)
        for r in reports_data:
            folder_id = r.get('final_folder_id')
            if folder_id:
                folder_assignments[folder_id] += 1
        
        if not folder_assignments:
            return {}
        
        assignment_counts = list(folder_assignments.values())
        assignment_counts_array = np.array(assignment_counts)
        
        # ジニ係数（不平等度）を計算
        gini = self._calculate_gini_coefficient(assignment_counts_array)
        
        # 変動係数（CV）
        mean_assignments = np.mean(assignment_counts_array)
        cv = np.std(assignment_counts_array) / mean_assignments if mean_assignments > 0 else 0
        
        return {
            'total_folders_used': len(folder_assignments),
            'mean_images_per_folder': float(mean_assignments),
            'std_images_per_folder': float(np.std(assignment_counts_array)),
            'min_images_per_folder': int(np.min(assignment_counts_array)),
            'max_images_per_folder': int(np.max(assignment_counts_array)),
            'gini_coefficient': float(gini),
            'coefficient_of_variation': float(cv),
            'balance_score': float(1 - gini),  # 0-1, 1が最もバランスが良い
            'folder_assignment_distribution': dict(folder_assignments)
        }
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """ジニ係数を計算（0: 完全平等, 1: 完全不平等）"""
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        # ジニ係数の計算
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        
        return float(gini)
    
    def _calculate_criteria_consistency(self, reports_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分類基準の一貫性を評価"""
        criteria_used = [r for r in reports_data if r.get('classification_criteria_used', False)]
        total_criteria_used = len(criteria_used)
        
        if total_criteria_used == 0:
            return {
                'criteria_used_count': 0,
                'consistency_score': 0.0,
                'note': '分類基準が使用されていません'
            }
        
        # 分類基準が使用された場合の成功率
        criteria_success = sum(
            1 for r in criteria_used
            if r.get('final_similarity', 0) >= 0.5  # 一定以上の類似度
        )
        
        consistency_score = criteria_success / total_criteria_used if total_criteria_used > 0 else 0
        
        return {
            'criteria_used_count': total_criteria_used,
            'criteria_success_count': criteria_success,
            'consistency_score': float(consistency_score),
            'consistency_percentage': float(consistency_score * 100)
        }
    
    def _calculate_confidence_scores(self, reports_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分類の信頼度スコアを計算"""
        # 信頼度の定義: 類似度が高く、エラーがない分類
        high_confidence_threshold = 0.7
        medium_confidence_threshold = 0.5
        
        high_confidence = sum(
            1 for r in reports_data
            if r.get('final_similarity', 0) >= high_confidence_threshold
            and len(r.get('errors', [])) == 0
        )
        
        medium_confidence = sum(
            1 for r in reports_data
            if medium_confidence_threshold <= r.get('final_similarity', 0) < high_confidence_threshold
            and len(r.get('errors', [])) == 0
        )
        
        low_confidence = len(reports_data) - high_confidence - medium_confidence
        
        total = len(reports_data)
        
        return {
            'high_confidence_count': high_confidence,
            'high_confidence_ratio': high_confidence / total if total > 0 else 0,
            'medium_confidence_count': medium_confidence,
            'medium_confidence_ratio': medium_confidence / total if total > 0 else 0,
            'low_confidence_count': low_confidence,
            'low_confidence_ratio': low_confidence / total if total > 0 else 0,
            'thresholds': {
                'high': high_confidence_threshold,
                'medium': medium_confidence_threshold
            }
        }
    
    def _calculate_performance_metrics(self, reports_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """パフォーマンス指標を計算"""
        # ベクトル取得成功率
        sentence_embedding_success = sum(
            1 for r in reports_data
            if r.get('sentence_embedding_available', False)
        )
        
        image_embedding_success = sum(
            1 for r in reports_data
            if r.get('image_embedding_available', False)
        )
        
        total = len(reports_data)
        
        return {
            'sentence_embedding_success_rate': sentence_embedding_success / total if total > 0 else 0,
            'image_embedding_success_rate': image_embedding_success / total if total > 0 else 0,
            'both_embeddings_available': sum(
                1 for r in reports_data
                if r.get('sentence_embedding_available', False) 
                and r.get('image_embedding_available', False)
            ),
            'both_embeddings_available_rate': sum(
                1 for r in reports_data
                if r.get('sentence_embedding_available', False) 
                and r.get('image_embedding_available', False)
            ) / total if total > 0 else 0
        }
    
    def _calculate_error_analysis(self, reports_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """エラー分析"""
        error_types = defaultdict(int)
        
        for r in reports_data:
            errors = r.get('errors', [])
            if errors:
                for error in errors:
                    # エラーメッセージからタイプを抽出
                    if '文章埋め込み' in error:
                        error_types['sentence_embedding_error'] += 1
                    elif '画像埋め込み' in error:
                        error_types['image_embedding_error'] += 1
                    elif 'フォルダ' in error:
                        error_types['folder_error'] += 1
                    else:
                        error_types['other_error'] += 1
        
        total_errors = sum(error_types.values())
        
        return {
            'total_errors': total_errors,
            'error_type_distribution': dict(error_types),
            'images_with_errors': sum(1 for r in reports_data if len(r.get('errors', [])) > 0),
            'error_free_ratio': sum(
                1 for r in reports_data if len(r.get('errors', [])) == 0
            ) / len(reports_data) if len(reports_data) > 0 else 0
        }
    
    def _calculate_hierarchy_quality(self, folder_data: Dict[str, Any]) -> Dict[str, Any]:
        """階層構造の品質を評価"""
        # この関数は folder_data の構造に依存するため、
        # 実際のデータ構造に合わせて実装が必要
        
        # 仮実装: 基本的な統計のみ
        return {
            'note': '階層構造の詳細分析は folder_data の構造に依存します',
            'folder_data_available': folder_data is not None and len(folder_data) > 0
        }
    
    def calculate_incremental_metrics(
        self,
        current_reports: List[Dict[str, Any]],
        previous_metrics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        前回との比較による増分メトリクスを計算
        
        Args:
            current_reports: 今回のレポートデータ
            previous_metrics: 前回の評価指標
            
        Returns:
            増分メトリクス
        """
        if not previous_metrics:
            return {'note': '前回のメトリクスが存在しないため、比較できません'}
        
        current_metrics = self.calculate_all_metrics(current_reports, {})
        
        improvements = {}
        
        # 基本統計の比較
        if 'basic_stats' in current_metrics and 'basic_stats' in previous_metrics:
            improvements['new_folder_ratio_change'] = (
                current_metrics['basic_stats']['new_folder_ratio'] -
                previous_metrics['basic_stats']['new_folder_ratio']
            )
        
        # 類似度の改善
        if 'similarity_stats' in current_metrics and 'similarity_stats' in previous_metrics:
            improvements['mean_similarity_change'] = (
                current_metrics['similarity_stats']['mean_similarity'] -
                previous_metrics['similarity_stats']['mean_similarity']
            )
        
        return {
            'improvements': improvements,
            'current_metrics': current_metrics,
            'previous_metrics': previous_metrics
        }
