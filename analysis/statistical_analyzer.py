import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any
import logging
from core.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Analisador estatístico avançado para dados"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    @memory_manager.memory_limiter
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise estatística completa do dataset"""
        logger.info(f"Iniciando análise estatística de dataset com {len(df)} linhas e {len(df.columns)} colunas")
        
        analysis = {
            'basic_stats': self._basic_statistics(df),
            'data_types': self._analyze_data_types(df),
            'missing_values': self._analyze_missing_values(df),
            'distributions': self._analyze_distributions(df),
            'correlations': self._analyze_correlations(df),
            'outliers': self._detect_outliers(df),
            'data_quality': self._assess_data_quality(df)
        }
        
        return analysis
    
    def _basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estatísticas básicas do dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        basic_stats = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(df.columns) - len(numeric_cols),
            'total_cells': df.shape[0] * df.shape[1],
            'non_null_cells': df.count().sum()
        }
        
        if len(numeric_cols) > 0:
            numeric_summary = df[numeric_cols].describe()
            basic_stats['numeric_summary'] = numeric_summary.to_dict()
        
        return basic_stats
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise detalhada dos tipos de dados"""
        type_analysis = {}
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                type_analysis[column] = {'type': 'empty', 'subtype': 'no_data'}
                continue
            
            dtype = str(df[column].dtype)
            
            # Análise específica por tipo
            if df[column].dtype in ['int64', 'float64']:
                subtype = self._analyze_numeric_type(col_data)
            elif df[column].dtype == 'object':
                subtype = self._analyze_object_type(col_data)
            elif df[column].dtype == 'datetime64[ns]':
                subtype = 'datetime'
            else:
                subtype = 'other'
            
            type_analysis[column] = {
                'type': dtype,
                'subtype': subtype,
                'unique_values': df[column].nunique(),
                'unique_ratio': df[column].nunique() / len(df),
                'sample_values': col_data.head(5).tolist()
            }
        
        return type_analysis
    
    def _analyze_numeric_type(self, series: pd.Series) -> str:
        """Analisa subtipo de dados numéricos"""
        # Verifica se são inteiros
        if series.dtype == 'int64' or (series % 1 == 0).all():
            if series.min() >= 0 and series.max() <= 1:
                return 'binary'
            elif series.nunique() < 10:
                return 'categorical_numeric'
            elif series.min() >= 0:
                return 'positive_integer'
            else:
                return 'integer'
        else:
            if series.min() >= 0 and series.max() <= 1:
                return 'probability'
            elif series.min() >= 0:
                return 'positive_float'
            else:
                return 'float'
    
    def _analyze_object_type(self, series: pd.Series) -> str:
        """Analisa subtipo de dados object"""
        sample = series.head(100)
        
        # Verifica se são datas
        try:
            pd.to_datetime(sample, errors='raise')
            return 'datetime_string'
        except:
            pass
        
        # Verifica se são números como string
        try:
            pd.to_numeric(sample, errors='raise')
            return 'numeric_string'
        except:
            pass
        
        # Verifica se é categórico
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.1:
            return 'categorical'
        elif unique_ratio < 0.5:
            return 'semi_categorical'
        else:
            # Analisa comprimento médio para determinar se é texto longo
            avg_length = series.astype(str).str.len().mean()
            if avg_length > 50:
                return 'long_text'
            else:
                return 'short_text'
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de valores faltantes"""
        missing_analysis = {}
        
        for column in df.columns:
            null_count = df[column].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            missing_analysis[column] = {
                'null_count': int(null_count),
                'null_percentage': float(null_percentage),
                'missing_pattern': self._analyze_missing_pattern(df[column])
            }
        
        # Análise global de padrões de falta
        total_missing = df.isnull().sum().sum()
        missing_analysis['overall'] = {
            'total_missing_values': int(total_missing),
            'missing_percentage': float((total_missing / (len(df) * len(df.columns))) * 100),
            'columns_with_missing': int((df.isnull().sum() > 0).sum()),
            'complete_rows': int(df.dropna().shape[0])
        }
        
        return missing_analysis
    
    def _analyze_missing_pattern(self, series: pd.Series) -> str:
        """Analisa padrão de valores faltantes"""
        if series.isnull().sum() == 0:
            return 'no_missing'
        elif series.isnull().sum() == len(series):
            return 'all_missing'
        else:
            # Verifica se há padrão sequencial
            null_positions = series.isnull()
            if null_positions.iloc[:len(series)//2].sum() > null_positions.iloc[len(series)//2:].sum() * 2:
                return 'missing_at_start'
            elif null_positions.iloc[len(series)//2:].sum() > null_positions.iloc[:len(series)//2].sum() * 2:
                return 'missing_at_end'
            else:
                return 'random_missing'
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de distribuições estatísticas"""
        distributions = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            data = df[column].dropna()
            
            if len(data) < 5:
                continue
            
            distributions[column] = {
                'normality_test': self._test_normality(data),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'distribution_type': self._identify_distribution(data),
                'quartiles': {
                    'Q1': float(data.quantile(0.25)),
                    'Q2': float(data.quantile(0.5)),
                    'Q3': float(data.quantile(0.75)),
                    'IQR': float(data.quantile(0.75) - data.quantile(0.25))
                }
            }
        
        return distributions
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Testa normalidade dos dados"""
        try:
            # Shapiro-Wilk para amostras pequenas
            if len(data) <= 5000:
                stat, p_value = stats.shapiro(data)
                test_name = 'shapiro_wilk'
            else:
                # Kolmogorov-Smirnov para amostras grandes
                stat, p_value = stats.kstest(data, 'norm')
                test_name = 'kolmogorov_smirnov'
            
            return {
                'test': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        except Exception as e:
            logger.warning(f"Erro no teste de normalidade: {e}")
            return {'test': 'failed', 'error': str(e)}
    
    def _identify_distribution(self, data: pd.Series) -> str:
        """Identifica tipo de distribuição"""
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Critérios simplificados
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        elif kurtosis < -1:
            return 'light_tailed'
        else:
            return 'unknown'
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de correlações"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        # Matriz de correlação
        corr_matrix = df[numeric_cols].corr()
        
        # Encontra correlações altas
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Threshold para correlação alta
                    high_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': self._classify_correlation_strength(abs(corr_value))
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'avg_correlation': float(np.mean(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]))),
            'max_correlation': float(np.max(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)])))
        }
    
    def _classify_correlation_strength(self, corr_value: float) -> str:
        """Classifica força da correlação"""
        if corr_value >= 0.9:
            return 'very_strong'
        elif corr_value >= 0.7:
            return 'strong'
        elif corr_value >= 0.5:
            return 'moderate'
        elif corr_value >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detecção de outliers"""
        outliers = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            data = df[column].dropna()
            
            if len(data) < 5:
                continue
            
            # Método IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Método Z-score
            z_scores = np.abs(stats.zscore(data))
            zscore_outliers = data[z_scores > 3]
            
            outliers[column] = {
                'iqr_method': {
                    'count': len(iqr_outliers),
                    'percentage': (len(iqr_outliers) / len(data)) * 100,
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                },
                'zscore_method': {
                    'count': len(zscore_outliers),
                    'percentage': (len(zscore_outliers) / len(data)) * 100
                }
            }
        
        return outliers
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Avaliação geral da qualidade dos dados"""
        # Calcula score de qualidade baseado em vários fatores
        factors = {}
        
        # Fator 1: Completude (ausência de valores faltantes)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_score = 1 - (missing_cells / total_cells)
        factors['completeness'] = float(completeness_score)
        
        # Fator 2: Consistência (tipos de dados apropriados)
        consistency_issues = 0
        for column in df.columns:
            if df[column].dtype == 'object':
                # Verifica se coluna numérica está como string
                sample = df[column].dropna().head(100)
                try:
                    pd.to_numeric(sample, errors='raise')
                    consistency_issues += 1
                except:
                    pass
        
        consistency_score = 1 - (consistency_issues / len(df.columns))
        factors['consistency'] = float(consistency_score)
        
        # Fator 3: Duplicação
        duplicate_rows = df.duplicated().sum()
        uniqueness_score = 1 - (duplicate_rows / len(df))
        factors['uniqueness'] = float(uniqueness_score)
        
        # Score geral (média ponderada)
        overall_score = (
            factors['completeness'] * 0.4 +
            factors['consistency'] * 0.3 +
            factors['uniqueness'] * 0.3
        )
        
        return {
            'overall_score': float(overall_score),
            'factors': factors,
            'quality_level': self._classify_quality_level(overall_score),
            'recommendations': self._generate_quality_recommendations(factors)
        }
    
    def _classify_quality_level(self, score: float) -> str:
        """Classifica nível de qualidade"""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_quality_recommendations(self, factors: Dict[str, float]) -> List[str]:
        """Gera recomendações para melhoria da qualidade"""
        recommendations = []
        
        if factors['completeness'] < 0.8:
            recommendations.append("Tratar valores faltantes - completude baixa")
        
        if factors['consistency'] < 0.8:
            recommendations.append("Verificar tipos de dados - inconsistências detectadas")
        
        if factors['uniqueness'] < 0.9:
            recommendations.append("Remover duplicatas - registros duplicados encontrados")
        
        if not recommendations:
            recommendations.append("Qualidade dos dados está adequada")
        
        return recommendations

# Instância global
statistical_analyzer = StatisticalAnalyzer()
