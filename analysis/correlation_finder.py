import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any, Optional
import logging
from core.memory_manager import memory_manager
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CorrelationFinder:
    """Analisador avançado de correlações e dependências entre variáveis"""
    
    def __init__(self):
        self.label_encoders = {}
        
    @memory_manager.memory_limiter
    def find_all_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Encontra todas as correlações possíveis no dataset"""
        logger.info("Iniciando análise completa de correlações")
        
        results = {
            'linear_correlations': self._find_linear_correlations(df),
            'nonlinear_correlations': self._find_nonlinear_correlations(df),
            'categorical_associations': self._find_categorical_associations(df),
            'mixed_correlations': self._find_mixed_correlations(df),
            'causal_relationships': self._detect_causal_relationships(df),
            'feature_importance': self._calculate_feature_importance(df),
            'correlation_networks': self._build_correlation_networks(df)
        }
        
        return results
    
    def _find_linear_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Encontra correlações lineares entre variáveis numéricas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insuficientes colunas numéricas para correlação linear'}
        
        correlations = {}
        
        # Matriz de correlação de Pearson
        pearson_matrix = df[numeric_cols].corr(method='pearson')
        
        # Matriz de correlação de Spearman (não-paramétrica)
        spearman_matrix = df[numeric_cols].corr(method='spearman')
        
        # Encontra correlações significativas
        significant_correlations = []
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                pearson_corr = pearson_matrix.loc[col1, col2]
                spearman_corr = spearman_matrix.loc[col1, col2]
                
                # Testa significância estatística
                data1 = df[col1].dropna()
                data2 = df[col2].dropna()
                
                # Alinha dados (remove NaN de ambas as séries)
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) < 3:
                    continue
                
                aligned_data1 = data1[common_idx]
                aligned_data2 = data2[common_idx]
                
                try:
                    # Teste de Pearson
                    pearson_stat, pearson_p = pearsonr(aligned_data1, aligned_data2)
                    
                    # Teste de Spearman
                    spearman_stat, spearman_p = spearmanr(aligned_data1, aligned_data2)
                    
                    # Teste de Kendall (para robustez)
                    kendall_stat, kendall_p = kendalltau(aligned_data1, aligned_data2)
                    
                except Exception as e:
                    logger.warning(f"Erro nos testes de correlação entre {col1} e {col2}: {e}")
                    continue
                
                # Considera significativa se p < 0.05 e correlação > 0.3
                if (abs(pearson_corr) > 0.3 and pearson_p < 0.05) or \
                   (abs(spearman_corr) > 0.3 and spearman_p < 0.05):
                    
                    correlation_strength = self._classify_correlation_strength(max(abs(pearson_corr), abs(spearman_corr)))
                    
                    significant_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'pearson_correlation': float(pearson_corr),
                        'pearson_p_value': float(pearson_p),
                        'spearman_correlation': float(spearman_corr),
                        'spearman_p_value': float(spearman_p),
                        'kendall_correlation': float(kendall_stat),
                        'kendall_p_value': float(kendall_p),
                        'strength': correlation_strength,
                        'sample_size': len(aligned_data1)
                    })
        
        correlations = {
            'pearson_matrix': pearson_matrix.to_dict(),
            'spearman_matrix': spearman_matrix.to_dict(),
            'significant_correlations': significant_correlations,
            'total_significant': len(significant_correlations)
        }
        
        return correlations
    
    def _find_nonlinear_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Encontra correlações não-lineares usando informação mútua"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insuficientes colunas numéricas para correlação não-linear'}
        
        nonlinear_correlations = []
        
        # Limita a análise para performance (máximo 10 colunas)
        cols_to_analyze = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
        
        for i, target_col in enumerate(cols_to_analyze):
            feature_cols = [col for col in cols_to_analyze if col != target_col]
            
            if not feature_cols:
                continue
            
            # Prepara dados
            target_data = df[target_col].dropna()
            feature_data = df[feature_cols].loc[target_data.index].fillna(target_data.mean())
            
            if len(target_data) < 10:
                continue
            
            try:
                # Calcula informação mútua
                mi_scores = mutual_info_regression(
                    feature_data, 
                    target_data, 
                    random_state=42
                )
                
                # Identifica correlações não-lineares significativas
                for j, feature_col in enumerate(feature_cols):
                    mi_score = mi_scores[j]
                    
                    if mi_score > 0.1:  # Threshold para significância
                        nonlinear_correlations.append({
                            'target_variable': target_col,
                            'feature_variable': feature_col,
                            'mutual_information': float(mi_score),
                            'nonlinear_strength': self._classify_mi_strength(mi_score),
                            'sample_size': len(target_data)
                        })
                        
            except Exception as e:
                logger.warning(f"Erro no cálculo de informação mútua para {target_col}: {e}")
                continue
        
        return {
            'nonlinear_correlations': nonlinear_correlations,
            'total_nonlinear': len(nonlinear_correlations)
        }
    
    def _find_categorical_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Encontra associações entre variáveis categóricas"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) < 2:
            return {'message': 'Insuficientes colunas categóricas para análise de associação'}
        
        associations = []
        
        # Limita análise para performance
        cols_to_analyze = categorical_cols[:8] if len(categorical_cols) > 8 else categorical_cols
        
        for i, col1 in enumerate(cols_to_analyze):
            for col2 in cols_to_analyze[i+1:]:
                try:
                    # Cria tabela de contingência
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    
                    # Teste qui-quadrado
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    # Calcula V de Cramér (medida de associação)
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                    
                    # Considera significativa se p < 0.05
                    if p_value < 0.05:
                        associations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'chi2_statistic': float(chi2),
                            'p_value': float(p_value),
                            'cramers_v': float(cramers_v),
                            'association_strength': self._classify_cramers_v_strength(cramers_v),
                            'degrees_of_freedom': int(dof),
                            'sample_size': int(n)
                        })
                        
                except Exception as e:
                    logger.warning(f"Erro na análise de associação entre {col1} e {col2}: {e}")
                    continue
        
        return {
            'categorical_associations': associations,
            'total_associations': len(associations)
        }
    
    def _find_mixed_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Encontra correlações entre variáveis numéricas e categóricas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) == 0 or len(categorical_cols) == 0:
            return {'message': 'Necessário pelo menos uma coluna numérica e uma categórica'}
        
        mixed_correlations = []
        
        # Limita análise para performance
        numeric_sample = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        categorical_sample = categorical_cols[:5] if len(categorical_cols) > 5 else categorical_cols
        
        for numeric_col in numeric_sample:
            for categorical_col in categorical_sample:
                try:
                    # Remove valores nulos
                    clean_data = df[[numeric_col, categorical_col]].dropna()
                    
                    if len(clean_data) < 10:
                        continue
                    
                    # Análise ANOVA (diferenças entre grupos)
                    groups = []
                    group_names = []
                    
                    for category in clean_data[categorical_col].unique():
                        group_data = clean_data[clean_data[categorical_col] == category][numeric_col]
                        if len(group_data) > 1:  # Precisa de pelo menos 2 observações
                            groups.append(group_data.values)
                            group_names.append(category)
                    
                    if len(groups) < 2:
                        continue
                    
                    # Teste ANOVA
                    f_statistic, anova_p_value = stats.f_oneway(*groups)
                    
                    # Eta squared (tamanho do efeito)
                    grand_mean = clean_data[numeric_col].mean()
                    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
                    ss_total = sum((clean_data[numeric_col] - grand_mean)**2)
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    # Informação mútua entre numérica e categórica
                    # Discretiza variável categórica
                    le = LabelEncoder()
                    categorical_encoded = le.fit_transform(clean_data[categorical_col])
                    mi_score = mutual_info_regression(
                        categorical_encoded.reshape(-1, 1),
                        clean_data[numeric_col],
                        random_state=42
                    )[0]
                    
                    if anova_p_value < 0.05 or mi_score > 0.1:
                        mixed_correlations.append({
                            'numeric_variable': numeric_col,
                            'categorical_variable': categorical_col,
                            'anova_f_statistic': float(f_statistic),
                            'anova_p_value': float(anova_p_value),
                            'eta_squared': float(eta_squared),
                            'mutual_information': float(mi_score),
                            'effect_size': self._classify_eta_squared(eta_squared),
                            'groups_analyzed': group_names[:5],  # Máximo 5 grupos
                            'sample_size': len(clean_data)
                        })
                        
                except Exception as e:
                    logger.warning(f"Erro na correlação mista entre {numeric_col} e {categorical_col}: {e}")
                    continue
        
        return {
            'mixed_correlations': mixed_correlations,
            'total_mixed': len(mixed_correlations)
        }
    
    def _detect_causal_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta possíveis relações causais (correlação não implica causalidade)"""
        # Esta é uma análise exploratória, não prova causalidade
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insuficientes colunas numéricas para análise causal'}
        
        potential_causal = []
        
        # Limita análise para performance
        cols_sample = numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
        
        for i, col1 in enumerate(cols_sample):
            for col2 in cols_sample[i+1:]:
                try:
                    # Dados alinhados
                    data = df[[col1, col2]].dropna()
                    
                    if len(data) < 10:
                        continue
                    
                    # Teste de Granger (causalidade temporal - simplificado)
                    # Aqui fazemos uma versão simplificada baseada em lag
                    if len(data) > 20:
                        data_sorted = data.sort_index()
                        
                        # Cria lag
                        data_lagged = data_sorted.copy()
                        data_lagged[f'{col1}_lag1'] = data_lagged[col1].shift(1)
                        data_lagged[f'{col2}_lag1'] = data_lagged[col2].shift(1)
                        data_lagged = data_lagged.dropna()
                        
                        if len(data_lagged) > 10:
                            # Correlação entre col1(t-1) e col2(t)
                            corr_1to2 = data_lagged[f'{col1}_lag1'].corr(data_lagged[col2])
                            
                            # Correlação entre col2(t-1) e col1(t)
                            corr_2to1 = data_lagged[f'{col2}_lag1'].corr(data_lagged[col1])
                            
                            # Direção causal baseada em qual lag correlation é maior
                            if abs(corr_1to2) > abs(corr_2to1) and abs(corr_1to2) > 0.3:
                                direction = f"{col1} → {col2}"
                                strength = abs(corr_1to2)
                            elif abs(corr_2to1) > abs(corr_1to2) and abs(corr_2to1) > 0.3:
                                direction = f"{col2} → {col1}"
                                strength = abs(corr_2to1)
                            else:
                                direction = "bidirecional/unclear"
                                strength = max(abs(corr_1to2), abs(corr_2to1))
                            
                            if strength > 0.3:
                                potential_causal.append({
                                    'relationship': f"{col1} ↔ {col2}",
                                    'suggested_direction': direction,
                                    'strength': float(strength),
                                    'lag1_correlation_1to2': float(corr_1to2),
                                    'lag1_correlation_2to1': float(corr_2to1),
                                    'sample_size': len(data_lagged),
                                    'note': 'Análise exploratória - não prova causalidade'
                                })
                    
                except Exception as e:
                    logger.warning(f"Erro na análise causal entre {col1} e {col2}: {e}")
                    continue
        
        return {
            'potential_causal_relationships': potential_causal,
            'total_potential_causal': len(potential_causal),
            'disclaimer': 'Esta análise é exploratória. Correlação não implica causalidade. '
                         'Análise causal adequada requer design experimental ou métodos econométricos específicos.'
        }
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula importância das features para cada variável numérica"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insuficientes colunas numéricas para análise de importância'}
        
        feature_importance = {}
        
        # Limita análise para performance
        cols_sample = numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
        
        for target_col in cols_sample:
            feature_cols = [col for col in cols_sample if col != target_col]
            
            if not feature_cols:
                continue
            
            try:
                # Prepara dados
                target_data = df[target_col].dropna()
                feature_data = df[feature_cols].loc[target_data.index]
                
                # Remove linhas com NaN em features
                complete_data = pd.concat([target_data, feature_data], axis=1).dropna()
                
                if len(complete_data) < 10:
                    continue
                
                target_clean = complete_data[target_col]
                features_clean = complete_data[feature_cols]
                
                # Informação mútua
                mi_scores = mutual_info_regression(
                    features_clean, 
                    target_clean, 
                    random_state=42
                )
                
                # Correlação absoluta
                corr_scores = [abs(features_clean[col].corr(target_clean)) for col in feature_cols]
                
                # Combina scores
                importance_scores = []
                for i, feature_col in enumerate(feature_cols):
                    combined_score = (mi_scores[i] + corr_scores[i]) / 2
                    importance_scores.append({
                        'feature': feature_col,
                        'mutual_information': float(mi_scores[i]),
                        'correlation_strength': float(corr_scores[i]),
                        'combined_importance': float(combined_score),
                        'importance_rank': 0  # Será preenchido depois
                    })
                
                # Ordena por importância
                importance_scores.sort(key=lambda x: x['combined_importance'], reverse=True)
                
                # Adiciona ranks
                for i, score in enumerate(importance_scores):
                    score['importance_rank'] = i + 1
                
                feature_importance[target_col] = importance_scores
                
            except Exception as e:
                logger.warning(f"Erro no cálculo de importância para {target_col}: {e}")
                continue
        
        return {
            'feature_importance_by_target': feature_importance,
            'targets_analyzed': len(feature_importance)
        }
    
    def _build_correlation_networks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Constrói redes de correlação para visualização"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 3:
            return {'message': 'Insuficientes colunas numéricas para rede de correlação'}
        
        # Matriz de correlação
        corr_matrix = df[numeric_cols].corr().abs()  # Valores absolutos
        
        # Constrói rede baseada em threshold
        threshold = 0.5
        network_edges = []
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                correlation = corr_matrix.loc[col1, col2]
                
                if correlation >= threshold and not np.isnan(correlation):
                    network_edges.append({
                        'source': col1,
                        'target': col2,
                        'weight': float(correlation),
                        'strength': self._classify_correlation_strength(correlation)
                    })
        
        # Métricas da rede
        nodes = list(numeric_cols)
        total_possible_edges = len(nodes) * (len(nodes) - 1) / 2
        density = len(network_edges) / total_possible_edges if total_possible_edges > 0 else 0
        
        # Centralidade dos nós (número de conexões)
        node_centrality = {}
        for node in nodes:
            connections = sum(1 for edge in network_edges 
                            if edge['source'] == node or edge['target'] == node)
            node_centrality[node] = connections
        
        return {
            'network_edges': network_edges,
            'network_nodes': nodes,
            'network_density': float(density),
            'total_edges': len(network_edges),
            'correlation_threshold': threshold,
            'node_centrality': node_centrality,
            'most_connected_variable': max(node_centrality, key=node_centrality.get) if node_centrality else None
        }
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classifica força da correlação"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _classify_mi_strength(self, mi_score: float) -> str:
        """Classifica força da informação mútua"""
        if mi_score >= 0.5:
            return 'very_strong'
        elif mi_score >= 0.3:
            return 'strong'
        elif mi_score >= 0.2:
            return 'moderate'
        elif mi_score >= 0.1:
            return 'weak'
        else:
            return 'very_weak'
    
    def _classify_cramers_v_strength(self, cramers_v: float) -> str:
        """Classifica força do V de Cramér"""
        if cramers_v >= 0.6:
            return 'very_strong'
        elif cramers_v >= 0.4:
            return 'strong'
        elif cramers_v >= 0.25:
            return 'moderate'
        elif cramers_v >= 0.1:
            return 'weak'
        else:
            return 'very_weak'
    
    def _classify_eta_squared(self, eta_squared: float) -> str:
        """Classifica tamanho do efeito (eta squared)"""
        if eta_squared >= 0.14:
            return 'large'
        elif eta_squared >= 0.06:
            return 'medium'
        elif eta_squared >= 0.01:
            return 'small'
        else:
            return 'negligible'

# Instância global
correlation_finder = CorrelationFinder()
