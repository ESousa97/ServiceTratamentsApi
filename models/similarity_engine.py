import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import logging
from models.embeddings import embedding_engine
from config.settings import config

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """Engine avançada para cálculo de similaridades semânticas"""
    
    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
    def calculate_column_similarities(self, df: pd.DataFrame, 
                                    text_columns: List[str]) -> Dict[str, Any]:
        """Calcula similaridades entre colunas de texto"""
        try:
            if len(text_columns) < 2:
                return {'message': 'Necessário pelo menos 2 colunas de texto'}
            
            similarities = {}
            column_embeddings = {}
            
            # Gera embeddings médios para cada coluna
            for column in text_columns:
                texts = df[column].dropna().astype(str).head(100).tolist()  # Amostra de 100
                
                if not texts:
                    continue
                
                embeddings = embedding_engine.generate_embeddings_batch(texts)
                avg_embedding = np.mean(embeddings, axis=0)
                column_embeddings[column] = avg_embedding
            
            # Calcula similaridades par a par
            for i, col1 in enumerate(text_columns):
                for col2 in text_columns[i+1:]:
                    if col1 in column_embeddings and col2 in column_embeddings:
                        similarity = cosine_similarity(
                            [column_embeddings[col1]], 
                            [column_embeddings[col2]]
                        )[0][0]
                        
                        similarities[f"{col1}_vs_{col2}"] = {
                            'similarity_score': float(similarity),
                            'similarity_level': self._classify_similarity_level(similarity),
                            'is_highly_similar': similarity > self.similarity_threshold
                        }
            
            # Análise geral
            if similarities:
                scores = [sim['similarity_score'] for sim in similarities.values()]
                analysis = {
                    'column_similarities': similarities,
                    'overall_statistics': {
                        'average_similarity': float(np.mean(scores)),
                        'max_similarity': float(np.max(scores)),
                        'min_similarity': float(np.min(scores)),
                        'highly_similar_pairs': sum(1 for sim in similarities.values() if sim['is_highly_similar'])
                    },
                    'recommendations': self._generate_similarity_recommendations(similarities)
                }
            else:
                analysis = {'message': 'Não foi possível calcular similaridades'}
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro no cálculo de similaridades entre colunas: {e}")
            return {'error': str(e)}
    
    def find_semantic_duplicates(self, texts: List[str], 
                               threshold: float = None) -> Dict[str, Any]:
        """Encontra duplicatas semânticas em lista de textos"""
        try:
            threshold = threshold or self.similarity_threshold
            
            if len(texts) < 2:
                return {'duplicates': [], 'unique_groups': []}
            
            # Limita tamanho para performance
            texts_sample = texts[:500] if len(texts) > 500 else texts
            
            # Gera embeddings
            embeddings = embedding_engine.generate_embeddings_batch(texts_sample)
            embeddings_array = np.array(embeddings)
            
            # Calcula matriz de similaridade
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Encontra pares similares
            duplicates = []
            processed_indices = set()
            
            for i in range(len(texts_sample)):
                if i in processed_indices:
                    continue
                
                similar_indices = []
                for j in range(i + 1, len(texts_sample)):
                    if j in processed_indices:
                        continue
                    
                    if similarity_matrix[i][j] >= threshold:
                        similar_indices.append(j)
                
                if similar_indices:
                    group = [i] + similar_indices
                    group_texts = [texts_sample[idx] for idx in group]
                    
                    duplicates.append({
                        'group_id': len(duplicates),
                        'indices': group,
                        'texts': group_texts[:5],  # Máximo 5 exemplos
                        'count': len(group),
                        'representative_text': texts_sample[i],
                        'avg_similarity': float(np.mean([similarity_matrix[i][j] for j in similar_indices]))
                    })
                    
                    processed_indices.update(group)
            
            # Textos únicos
            unique_indices = [i for i in range(len(texts_sample)) if i not in processed_indices]
            unique_texts = [texts_sample[i] for i in unique_indices]
            
            return {
                'duplicates': duplicates,
                'unique_groups': len(duplicates),
                'unique_texts_count': len(unique_texts),
                'total_processed': len(texts_sample),
                'duplicate_ratio': len(processed_indices) / len(texts_sample) if texts_sample else 0
            }
            
        except Exception as e:
            logger.error(f"Erro na busca de duplicatas semânticas: {e}")
            return {'duplicates': [], 'error': str(e)}
    
    def cluster_by_similarity(self, texts: List[str], 
                            eps: float = None, min_samples: int = 2) -> Dict[str, Any]:
        """Agrupa textos por similaridade usando clustering"""
        try:
            if len(texts) < min_samples:
                return {'clusters': {}, 'noise': texts}
            
            eps = eps or (1 - self.similarity_threshold)
            
            # Limita tamanho para performance
            texts_sample = texts[:1000] if len(texts) > 1000 else texts
            
            # Gera embeddings
            embeddings = embedding_engine.generate_embeddings_batch(texts_sample)
            embeddings_array = np.array(embeddings)
            
            # Clustering com DBSCAN
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='cosine'
            ).fit(embeddings_array)
            
            # Organiza resultados
            clusters = {}
            noise = []
            
            for i, label in enumerate(clustering.labels_):
                if label == -1:  # Ruído
                    noise.append(texts_sample[i])
                else:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(texts_sample[i])
            
            # Análise dos clusters
            cluster_analysis = {}
            for cluster_id, cluster_texts in clusters.items():
                # Calcula coesão do cluster
                cluster_indices = [i for i, label in enumerate(clustering.labels_) if label == cluster_id]
                cluster_embeddings = embeddings_array[cluster_indices]
                
                if len(cluster_embeddings) > 1:
                    intra_similarities = cosine_similarity(cluster_embeddings)
                    # Remove diagonal
                    mask = np.ones_like(intra_similarities, dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg_intra_similarity = np.mean(intra_similarities[mask])
                else:
                    avg_intra_similarity = 1.0
                
                cluster_analysis[cluster_id] = {
                    'texts': cluster_texts[:10],  # Máximo 10 exemplos
                    'size': len(cluster_texts),
                    'cohesion': float(avg_intra_similarity),
                    'representative_text': cluster_texts[0]
                }
            
            return {
                'clusters': cluster_analysis,
                'noise': noise[:10],  # Máximo 10 exemplos de ruído
                'total_clusters': len(clusters),
                'noise_count': len(noise),
                'clustered_ratio': (len(texts_sample) - len(noise)) / len(texts_sample) if texts_sample else 0
            }
            
        except Exception as e:
            logger.error(f"Erro no clustering por similaridade: {e}")
            return {'clusters': {}, 'error': str(e)}
    
    def find_most_similar_texts(self, query_text: str, 
                              candidate_texts: List[str], 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Encontra textos mais similares a um texto de consulta"""
        try:
            if not candidate_texts:
                return []
            
            # Limita candidatos para performance
            candidates_sample = candidate_texts[:1000] if len(candidate_texts) > 1000 else candidate_texts
            
            # Gera embeddings
            query_embedding = embedding_engine.generate_embedding(query_text)
            candidate_embeddings = embedding_engine.generate_embeddings_batch(candidates_sample)
            
            # Calcula similaridades
            similarities = cosine_similarity(
                [query_embedding], 
                candidate_embeddings
            )[0]
            
            # Ordena por similaridade
            sorted_indices = np.argsort(similarities)[::-1]
            
            results = []
            for i in sorted_indices[:top_k]:
                similarity_score = similarities[i]
                
                if similarity_score >= self.similarity_threshold:
                    results.append({
                        'text': candidates_sample[i],
                        'similarity_score': float(similarity_score),
                        'similarity_level': self._classify_similarity_level(similarity_score),
                        'rank': len(results) + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na busca de textos similares: {e}")
            return []
    
    def calculate_text_diversity(self, texts: List[str]) -> Dict[str, Any]:
        """Calcula diversidade semântica de uma coleção de textos"""
        try:
            if len(texts) < 2:
                return {'diversity_score': 0, 'message': 'Dados insuficientes'}
            
            # Limita tamanho para performance
            texts_sample = texts[:200] if len(texts) > 200 else texts
            
            # Gera embeddings
            embeddings = embedding_engine.generate_embeddings_batch(texts_sample)
            embeddings_array = np.array(embeddings)
            
            # Calcula matriz de similaridade
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Remove diagonal (auto-similaridade)
            mask = np.ones_like(similarity_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            similarities = similarity_matrix[mask]
            
            # Métricas de diversidade
            avg_similarity = np.mean(similarities)
            diversity_score = 1 - avg_similarity  # Diversidade = 1 - Similaridade média
            
            # Análise adicional
            high_similarity_pairs = np.sum(similarities > 0.8)
            total_pairs = len(similarities)
            
            analysis = {
                'diversity_score': float(diversity_score),
                'diversity_level': self._classify_diversity_level(diversity_score),
                'average_similarity': float(avg_similarity),
                'similarity_std': float(np.std(similarities)),
                'high_similarity_pairs': int(high_similarity_pairs),
                'high_similarity_ratio': float(high_similarity_pairs / total_pairs) if total_pairs > 0 else 0,
                'sample_size': len(texts_sample)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro no cálculo de diversidade: {e}")
            return {'diversity_score': 0, 'error': str(e)}
    
    def analyze_semantic_coherence(self, texts: List[str]) -> Dict[str, Any]:
        """Analisa coerência semântica de uma coleção de textos"""
        try:
            if len(texts) < 3:
                return {'coherence_score': 0, 'message': 'Dados insuficientes'}
            
            # Limita tamanho para performance
            texts_sample = texts[:100] if len(texts) > 100 else texts
            
            # Gera embeddings
            embeddings = embedding_engine.generate_embeddings_batch(texts_sample)
            embeddings_array = np.array(embeddings)
            
            # Calcula centroide
            centroid = np.mean(embeddings_array, axis=0)
            
            # Calcula distâncias ao centroide
            distances = [
                1 - cosine_similarity([embedding], [centroid])[0][0]
                for embedding in embeddings_array
            ]
            
            # Métricas de coerência
            avg_distance = np.mean(distances)
            coherence_score = 1 - avg_distance  # Coerência = 1 - Distância média ao centroide
            
            # Identifica outliers semânticos
            distance_threshold = np.mean(distances) + 2 * np.std(distances)
            outlier_indices = [i for i, d in enumerate(distances) if d > distance_threshold]
            
            analysis = {
                'coherence_score': float(coherence_score),
                'coherence_level': self._classify_coherence_level(coherence_score),
                'average_distance_to_centroid': float(avg_distance),
                'distance_std': float(np.std(distances)),
                'semantic_outliers': {
                    'count': len(outlier_indices),
                    'indices': outlier_indices[:5],  # Máximo 5 exemplos
                    'texts': [texts_sample[i] for i in outlier_indices[:5]]
                },
                'sample_size': len(texts_sample)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro na análise de coerência semântica: {e}")
            return {'coherence_score': 0, 'error': str(e)}
    
    def _classify_similarity_level(self, similarity: float) -> str:
        """Classifica nível de similaridade"""
        if similarity >= 0.9:
            return 'very_high'
        elif similarity >= 0.7:
            return 'high'
        elif similarity >= 0.5:
            return 'moderate'
        elif similarity >= 0.3:
            return 'low'
        else:
            return 'very_low'
    
    def _classify_diversity_level(self, diversity: float) -> str:
        """Classifica nível de diversidade"""
        if diversity >= 0.8:
            return 'very_diverse'
        elif diversity >= 0.6:
            return 'diverse'
        elif diversity >= 0.4:
            return 'moderate'
        elif diversity >= 0.2:
            return 'low_diversity'
        else:
            return 'very_low_diversity'
    
    def _classify_coherence_level(self, coherence: float) -> str:
        """Classifica nível de coerência"""
        if coherence >= 0.8:
            return 'very_coherent'
        elif coherence >= 0.6:
            return 'coherent'
        elif coherence >= 0.4:
            return 'moderate'
        elif coherence >= 0.2:
            return 'low_coherence'
        else:
            return 'incoherent'
    
    def _generate_similarity_recommendations(self, similarities: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na análise de similaridade"""
        recommendations = []
        
        if not similarities:
            return recommendations
        
        # Analisa similaridades altas
        high_similarity_pairs = [
            pair for pair, data in similarities.items() 
            if data['is_highly_similar']
        ]
        
        if len(high_similarity_pairs) > 0:
            recommendations.append(
                f"Encontradas {len(high_similarity_pairs)} pares de colunas altamente similares - "
                "considere consolidação ou remoção de redundâncias"
            )
        
        # Analisa similaridades baixas
        scores = [sim['similarity_score'] for sim in similarities.values()]
        avg_score = np.mean(scores)
        
        if avg_score < 0.3:
            recommendations.append(
                "Baixa similaridade entre colunas - dados parecem ser complementares"
            )
        elif avg_score > 0.8:
            recommendations.append(
                "Alta similaridade geral - muitas colunas contêm informações redundantes"
            )
        else:
            recommendations.append(
                "Similaridade moderada entre colunas - estrutura de dados equilibrada"
            )
        
        return recommendations

# Instância global
similarity_engine = SimilarityEngine()
