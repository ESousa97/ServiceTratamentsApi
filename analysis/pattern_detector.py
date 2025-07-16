import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import re
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from core.memory_manager import memory_manager
from scipy import stats

logger = logging.getLogger(__name__)

class PatternDetector:
    """Detector de padrões avançado para dados"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    @memory_manager.memory_limiter
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta todos os tipos de padrões no dataset"""
        logger.info("Iniciando detecção de padrões completa")
        patterns = {
            'temporal_patterns': self._detect_temporal_patterns(df),
            'clustering_patterns': self._detect_clustering_patterns(df),
            'sequence_patterns': self._detect_sequence_patterns(df),
            'anomaly_patterns': self._detect_anomalies(df),
            'text_patterns': self._detect_text_patterns(df),
            'numerical_patterns': self._detect_numerical_patterns(df),
            'categorical_patterns': self._detect_categorical_patterns(df)
        }
        return patterns

    def _detect_clustering_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta padrões de clustering nos dados numéricos - VERSÃO ROBUSTA"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'message': 'Insufficient numeric columns for clustering'}

        try:
            numeric_data = df[numeric_cols].copy()
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            numeric_data = numeric_data.dropna(axis=1, how='all')

            if numeric_data.empty or len(numeric_data.columns) < 2:
                return {'message': 'No valid numeric data for clustering after cleaning'}

            threshold = len(numeric_data.columns) * 0.5
            numeric_data = numeric_data.dropna(thresh=threshold)

            if len(numeric_data) < 10:
                return {'message': 'Insufficient data points for clustering after cleaning'}

            numeric_data = numeric_data.fillna(numeric_data.median())

            if not np.isfinite(numeric_data.values).all():
                finite_mask = np.isfinite(numeric_data.values).all(axis=0)
                numeric_data = numeric_data.loc[:, finite_mask]
                if numeric_data.empty or len(numeric_data.columns) < 2:
                    return {'message': 'No columns with finite values for clustering'}

            if len(numeric_data) > 5000:
                numeric_data = numeric_data.sample(n=5000, random_state=42)
                logger.info(f"Dados reduzidos para 5000 amostras para clustering")

            if len(numeric_data.columns) > 10:
                variances = numeric_data.var().sort_values(ascending=False)
                top_cols = variances.head(10).index
                numeric_data = numeric_data[top_cols]
                logger.info(f"Clustering limitado a {len(top_cols)} colunas com maior variância")

            data_array = numeric_data.values
            if np.all(data_array.std(axis=0) < 1e-10):
                return {'message': 'Data has insufficient variance for clustering'}

            try:
                scaled_data = self.scaler.fit_transform(data_array)
                if not np.isfinite(scaled_data).all():
                    scaled_data = (data_array - np.nanmean(data_array, axis=0)) / (np.nanstd(data_array, axis=0) + 1e-8)
                    scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=1.0, neginf=-1.0)
            except Exception as scaling_error:
                logger.warning(f"Erro no scaling: {scaling_error}")
                scaled_data = (data_array - np.nanmean(data_array, axis=0)) / (np.nanstd(data_array, axis=0) + 1e-8)
                scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=1.0, neginf=-1.0)

            clustering_results = {}
            algorithms = {
                'kmeans': self._apply_kmeans_clustering_safe,
                'dbscan': self._apply_dbscan_clustering_safe,
                'hierarchical': self._apply_hierarchical_clustering_safe
            }

            for alg_name, alg_func in algorithms.items():
                try:
                    result = alg_func(scaled_data)
                    clustering_results[alg_name] = result
                except Exception as e:
                    logger.warning(f"Erro no clustering {alg_name}: {e}")
                    clustering_results[alg_name] = {'error': str(e)}

            clustering_results['data_cleaning_info'] = {
                'original_shape': df[numeric_cols].shape,
                'cleaned_shape': numeric_data.shape,
                'columns_used': numeric_data.columns.tolist(),
                'samples_used': len(numeric_data)
            }
            return clustering_results

        except Exception as e:
            logger.error(f"Erro geral na detecção de clustering: {e}")
            return {'error': f'Clustering analysis failed: {str(e)}'}

    def _apply_kmeans_clustering_safe(self, data: np.ndarray) -> Dict[str, Any]:
        try:
            if data.shape[0] < 4:
                return {'error': 'Insufficient samples for K-means (need at least 4)'}
            max_k = min(8, len(data) // 3)
            if max_k < 2:
                return {'error': 'Cannot determine optimal number of clusters'}
            silhouette_scores = []
            for k in range(2, max_k + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
                    labels = kmeans.fit_predict(data)
                    if len(np.unique(labels)) < 2:
                        continue
                    score = silhouette_score(data, labels)
                    silhouette_scores.append((k, score))
                except Exception as e:
                    logger.warning(f"K-means falhou para k={k}: {e}")
                    continue
            if not silhouette_scores:
                return {'error': 'K-means clustering failed for all k values'}
            best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=100)
            labels = kmeans.fit_predict(data)
            return {
                'optimal_clusters': best_k,
                'silhouette_score': float(best_score),
                'cluster_sizes': [int(np.sum(labels == i)) for i in range(best_k)],
                'inertia': float(getattr(kmeans, 'inertia_', 0.0))
            }
        except Exception as e:
            return {'error': f'K-means clustering failed: {str(e)}'}

    def _apply_dbscan_clustering_safe(self, data: np.ndarray) -> Dict[str, Any]:
        try:
            if data.shape[0] < 10:
                return {'error': 'Insufficient samples for DBSCAN (need at least 10)'}
            min_samples = max(3, min(10, len(data) // 20))
            eps = 0.5
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            return {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'noise_ratio': float(n_noise / len(labels)) if len(labels) > 0 else 0.0,
                'cluster_sizes': [int(np.sum(labels == i)) for i in set(labels) if i != -1],
                'eps_used': eps,
                'min_samples_used': min_samples
            }
        except Exception as e:
            return {'error': f'DBSCAN clustering failed: {str(e)}'}

    def _apply_hierarchical_clustering_safe(self, data: np.ndarray) -> Dict[str, Any]:
        try:
            if data.shape[0] < 4:
                return {'error': 'Insufficient samples for hierarchical clustering (need at least 4)'}
            max_clusters = min(8, len(data) // 3)
            if max_clusters < 2:
                return {'error': 'Cannot determine number of clusters for hierarchical'}
            hierarchical = AgglomerativeClustering(n_clusters=max_clusters)
            labels = hierarchical.fit_predict(data)
            if len(set(labels)) > 1:
                silhouette = silhouette_score(data, labels)
            else:
                silhouette = 0.0
            return {
                'n_clusters': len(set(labels)),
                'silhouette_score': float(silhouette),
                'cluster_sizes': [int(np.sum(labels == i)) for i in set(labels)]
            }
        except Exception as e:
            return {'error': f'Hierarchical clustering failed: {str(e)}'}

    def _detect_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        temporal_patterns = {}
        date_columns = []
        for column in df.columns:
            if df[column].dtype == 'datetime64[ns]':
                date_columns.append(column)
            elif df[column].dtype == 'object':
                sample = df[column].dropna().head(100)
                try:
                    pd.to_datetime(sample, errors='raise')
                    date_columns.append(column)
                except:
                    continue
        for date_col in date_columns:
            try:
                if df[date_col].dtype != 'datetime64[ns]':
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                date_data = df[date_col].dropna()
                if len(date_data) < 2:
                    continue
                date_range = date_data.max() - date_data.min()
                date_counts = date_data.value_counts().sort_index()
                seasonality = self._detect_seasonality(date_data)
                trends = self._detect_trends(date_data, date_counts)
                intervals = self._detect_regular_intervals(date_data)
                temporal_patterns[date_col] = {
                    'date_range': {
                        'start': date_data.min().isoformat(),
                        'end': date_data.max().isoformat(),
                        'span_days': date_range.days
                    },
                    'frequency_analysis': {
                        'most_common_date': date_counts.index[0].isoformat(),
                        'max_frequency': int(date_counts.iloc[0]),
                        'unique_dates': len(date_counts)
                    },
                    'seasonality': seasonality,
                    'trends': trends,
                    'regular_intervals': intervals
                }
            except Exception as e:
                logger.warning(f"Erro na análise temporal da coluna {date_col}: {e}")
                temporal_patterns[date_col] = {'error': str(e)}
        return temporal_patterns

    def _detect_seasonality(self, date_series: pd.Series) -> Dict[str, Any]:
        seasonality = {}
        weekday_counts = date_series.dt.dayofweek.value_counts().sort_index()
        weekday_pattern = weekday_counts.std() / weekday_counts.mean() if weekday_counts.mean() > 0 else 0
        month_counts = date_series.dt.month.value_counts().sort_index()
        monthly_pattern = month_counts.std() / month_counts.mean() if month_counts.mean() > 0 else 0
        if date_series.dt.hour.nunique() > 1:
            hour_counts = date_series.dt.hour.value_counts().sort_index()
            hourly_pattern = hour_counts.std() / hour_counts.mean() if hour_counts.mean() > 0 else 0
        else:
            hourly_pattern = 0
        seasonality = {
            'weekday_seasonality': float(weekday_pattern),
            'monthly_seasonality': float(monthly_pattern),
            'hourly_seasonality': float(hourly_pattern),
            'has_strong_weekly_pattern': weekday_pattern > 0.5,
            'has_strong_monthly_pattern': monthly_pattern > 0.5,
            'has_strong_hourly_pattern': hourly_pattern > 0.5
        }
        return seasonality

    def _detect_trends(self, date_series: pd.Series, date_counts: pd.Series) -> Dict[str, Any]:
        if len(date_counts) < 3:
            return {'trend': 'insufficient_data'}
        x = np.arange(len(date_counts))
        y = date_counts.values
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        trend_type = 'stable'
        if correlation > 0.3:
            trend_type = 'increasing'
        elif correlation < -0.3:
            trend_type = 'decreasing'
        return {
            'trend_correlation': float(correlation),
            'trend_type': trend_type,
            'trend_strength': abs(float(correlation))
        }

    def _detect_regular_intervals(self, date_series: pd.Series) -> Dict[str, Any]:
        if len(date_series) < 3:
            return {'regular_intervals': False}
        sorted_dates = date_series.sort_values()
        intervals = sorted_dates.diff().dropna()
        if len(intervals) == 0:
            return {'regular_intervals': False}
        most_common_interval = intervals.mode()
        if len(most_common_interval) > 0:
            interval_days = most_common_interval.iloc[0].days
            interval_consistency = (intervals == most_common_interval.iloc[0]).mean()
        else:
            interval_days = 0
            interval_consistency = 0
        return {
            'regular_intervals': interval_consistency > 0.7,
            'common_interval_days': interval_days,
            'interval_consistency': float(interval_consistency)
        }

    def _detect_sequence_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        sequence_patterns = {}
        for column in df.columns:
            if df[column].dtype in ['object', 'category']:
                patterns = self._analyze_categorical_sequences(df[column])
            elif np.issubdtype(df[column].dtype, np.number):
                patterns = self._analyze_numerical_sequences(df[column])
            else:
                continue
            if patterns:
                sequence_patterns[column] = patterns
        return sequence_patterns

    def _analyze_categorical_sequences(self, series: pd.Series) -> Dict[str, Any]:
        values = series.dropna().astype(str).tolist()
        if len(values) < 3:
            return {}
        patterns = {}
        bigrams = [(values[i], values[i+1]) for i in range(len(values)-1)]
        bigram_counts = pd.Series(bigrams).value_counts()
        trigrams = [(values[i], values[i+1], values[i+2]) for i in range(len(values)-2)]
        trigram_counts = pd.Series(trigrams).value_counts()
        return {
            'most_common_bigram': str(bigram_counts.index[0]) if len(bigram_counts) > 0 else None,
            'bigram_frequency': int(bigram_counts.iloc[0]) if len(bigram_counts) > 0 else 0,
            'most_common_trigram': str(trigram_counts.index[0]) if len(trigram_counts) > 0 else None,
            'trigram_frequency': int(trigram_counts.iloc[0]) if len(trigram_counts) > 0 else 0,
            'unique_bigrams': len(bigram_counts),
            'unique_trigrams': len(trigram_counts)
        }

    def _analyze_numerical_sequences(self, series: pd.Series) -> Dict[str, Any]:
        values = series.dropna().values
        if len(values) < 3:
            return {}
        differences = np.diff(values)
        diff_std = np.std(differences) if len(differences) > 1 else float('inf')
        is_arithmetic = diff_std < 1e-6
        if not np.any(values == 0):
            ratios = values[1:] / values[:-1]
            ratio_std = np.std(ratios) if len(ratios) > 1 else float('inf')
            is_geometric = ratio_std < 1e-6
        else:
            is_geometric = False
            ratios = []
        return {
            'is_arithmetic_sequence': bool(is_arithmetic),
            'arithmetic_difference': float(np.mean(differences)) if len(differences) > 0 else 0,
            'is_geometric_sequence': bool(is_geometric),
            'geometric_ratio': float(np.mean(ratios)) if len(ratios) > 0 else 0,
            'sequence_volatility': float(diff_std)
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {'message': 'No numeric columns for anomaly detection'}
        anomalies = {}
        try:
            numeric_data = df[numeric_cols].copy()
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            numeric_data = numeric_data.dropna(axis=1, thresh=len(numeric_data) * 0.1)
            if numeric_data.empty:
                return {'message': 'No valid data for anomaly detection after cleaning'}
            numeric_data = numeric_data.fillna(numeric_data.median())
            if len(numeric_data) > 10000:
                numeric_data = numeric_data.sample(n=10000, random_state=42)
            try:
                isolation_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    max_samples='auto',
                    max_features=1.0
                )
                data_array = numeric_data.values
                if not np.isfinite(data_array).all():
                    data_array = np.nan_to_num(data_array, nan=0.0, posinf=1.0, neginf=-1.0)
                anomaly_labels = isolation_forest.fit_predict(data_array)
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                anomaly_count = len(anomaly_indices)
                anomalies['isolation_forest'] = {
                    'anomaly_count': anomaly_count,
                    'anomaly_percentage': float((anomaly_count / len(numeric_data)) * 100),
                    'anomaly_indices': anomaly_indices.tolist()[:10],
                    'total_samples_analyzed': len(numeric_data)
                }
            except Exception as e:
                logger.warning(f"Erro no Isolation Forest: {e}")
                anomalies['isolation_forest'] = {'error': str(e)}
            for column in numeric_data.columns:
                try:
                    col_data = numeric_data[column].dropna()
                    if len(col_data) < 3:
                        continue
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    if std_val == 0 or not np.isfinite(std_val):
                        continue
                    z_scores = np.abs((col_data - mean_val) / std_val)
                    anomaly_mask = z_scores > 3
                    anomalies[f'{column}_zscore'] = {
                        'anomaly_count': int(anomaly_mask.sum()),
                        'anomaly_percentage': float((anomaly_mask.sum() / len(col_data)) * 100),
                        'max_zscore': float(z_scores.max()) if len(z_scores) > 0 else 0.0
                    }
                except Exception as e:
                    logger.warning(f"Erro no Z-score para {column}: {e}")
                    anomalies[f'{column}_zscore'] = {'error': str(e)}
            return anomalies
        except Exception as e:
            logger.error(f"Erro geral na detecção de anomalias: {e}")
            return {'error': f'Anomaly detection failed: {str(e)}'}

    def _detect_text_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        text_patterns = {}
        text_cols = df.select_dtypes(include=['object']).columns
        for column in text_cols:
            text_data = df[column].dropna().astype(str)
            if len(text_data) == 0:
                continue
            patterns = {
                'format_patterns': self._detect_format_patterns(text_data),
                'length_patterns': self._detect_length_patterns(text_data),
                'content_patterns': self._detect_content_patterns(text_data)
            }
            text_patterns[column] = patterns
        return text_patterns

    def _detect_format_patterns(self, text_series: pd.Series) -> Dict[str, Any]:
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[\d\s\-\(\)]{10,}$',
            'cpf': r'^\d{3}\.\d{3}\.\d{3}-\d{2}$',
            'cnpj': r'^\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}$',
            'cep': r'^\d{5}-?\d{3}$',
            'date': r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{4}$',
            'number': r'^\d+$',
            'decimal': r'^\d+[,\.]\d+$'
        }
        format_matches = {}
        for pattern_name, regex in patterns.items():
            matches = text_series.str.match(regex, na=False).sum()
            if matches > 0:
                format_matches[pattern_name] = {
                    'count': int(matches),
                    'percentage': float((matches / len(text_series)) * 100)
                }
        return format_matches

    def _detect_length_patterns(self, text_series: pd.Series) -> Dict[str, Any]:
        lengths = text_series.str.len()
        return {
            'avg_length': float(lengths.mean()),
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'std_length': float(lengths.std()),
            'common_lengths': lengths.value_counts().head(5).to_dict()
        }

    def _detect_content_patterns(self, text_series: pd.Series) -> Dict[str, Any]:
        special_chars = text_series.str.count(r'[^a-zA-Z0-9\s]')
        uppercase_ratio = text_series.str.count(r'[A-Z]') / text_series.str.len()
        return {
            'avg_special_chars': float(special_chars.mean()),
            'avg_uppercase_ratio': float(uppercase_ratio.mean()),
            'contains_numbers': int(text_series.str.contains(r'\d', na=False).sum()),
            'all_uppercase': int(text_series.str.isupper().sum()),
            'all_lowercase': int(text_series.str.islower().sum())
        }

    def _detect_numerical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numerical_patterns = {}
        for column in numeric_cols:
            data = df[column].dropna()
            if len(data) == 0:
                continue
            patterns = {
                'value_patterns': self._analyze_value_patterns(data),
                'distribution_patterns': self._analyze_distribution_patterns(data),
                'range_patterns': self._analyze_range_patterns(data)
            }
            numerical_patterns[column] = patterns
        return numerical_patterns

    def _analyze_value_patterns(self, series: pd.Series) -> Dict[str, Any]:
        values = series.values
        is_integer = np.all(values == np.round(values))
        common_multiples = {}
        for base in [2, 5, 10, 100]:
            multiple_count = np.sum(values % base == 0)
            if multiple_count > len(values) * 0.3:
                common_multiples[base] = int(multiple_count)
        return {
            'all_integers': bool(is_integer),
            'common_multiples': common_multiples,
            'zero_count': int(np.sum(values == 0)),
            'negative_count': int(np.sum(values < 0)),
            'unique_values': len(np.unique(values))
        }

    def _analyze_distribution_patterns(self, series: pd.Series) -> Dict[str, Any]:
        values = series.values
        distributions = {
            'normal': stats.normaltest(values)[1] > 0.05 if len(values) > 8 else False,
            'uniform': stats.kstest(values, 'uniform')[1] > 0.05 if len(values) > 8 else False,
            'exponential': stats.kstest(values, 'expon')[1] > 0.05 if len(values) > 8 else False
        }
        return {
            'distribution_tests': distributions,
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values)),
            'is_symmetric': abs(stats.skew(values)) < 0.5
        }

    def _analyze_range_patterns(self, series: pd.Series) -> Dict[str, Any]:
        values = series.values
        percentiles = np.percentile(values, [10, 25, 50, 75, 90])
        return {
            'range': float(values.max() - values.min()),
            'iqr': float(percentiles[3] - percentiles[1]),
            'percentiles': {
                'p10': float(percentiles[0]),
                'p25': float(percentiles[1]),
                'p50': float(percentiles[2]),
                'p75': float(percentiles[3]),
                'p90': float(percentiles[4])
            }
        }

    def _detect_categorical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        categorical_patterns = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_cols:
            data = df[column].dropna()
            if len(data) == 0:
                continue
            value_counts = data.value_counts()
            patterns = {
                'frequency_distribution': {
                    'most_common': str(value_counts.index[0]),
                    'most_common_count': int(value_counts.iloc[0]),
                    'least_common': str(value_counts.index[-1]),
                    'least_common_count': int(value_counts.iloc[-1]),
                    'unique_values': len(value_counts)
                },
                'concentration': {
                    'top_10_concentration': float(value_counts.head(10).sum() / len(data) * 100),
                    'singleton_count': int((value_counts == 1).sum()),
                    'gini_coefficient': self._calculate_gini_coefficient(value_counts.values)
                }
            }
            categorical_patterns[column] = patterns
        return categorical_patterns

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n) if cumsum[-1] > 0 else 0

# Instância global
pattern_detector = PatternDetector()
