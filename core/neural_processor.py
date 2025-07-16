import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from core.memory_manager import memory_manager
from models.embeddings import EmbeddingEngine
from models.neural_network import TextClassifier, AnomalyDetector
from models.similarity_engine import SimilarityEngine

logger = logging.getLogger(__name__)

def get_embedding_engine():
    """Obtém a instância global do embedding engine ou cria uma nova"""
    try:
        import builtins
        engine = getattr(builtins, 'global_embedding_engine', None)
        if engine is not None:
            return engine
    except:
        pass
    
    # Fallback: cria nova instância
    return EmbeddingEngine()

class NeuralProcessor:
    """Coordenador de processamento neural dos dados"""
    
    def __init__(self, embedding_engine: EmbeddingEngine = None):
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.text_classifier = TextClassifier(self.embedding_engine)
        self.anomaly_detector = AnomalyDetector()
        self.similarity_engine = SimilarityEngine(embedding_engine=self.embedding_engine)
        
    @memory_manager.memory_limiter
    def process_dataset_neural(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Processa dataset completo com redes neurais"""
        logger.info("Iniciando processamento neural do dataset")
        
        results = {
            'text_classification': self._process_text_classification(df),
            'anomaly_detection': self._process_anomaly_detection(df),
            'similarity_analysis': self._process_similarity_analysis(df),
            'pattern_recognition': self._process_pattern_recognition(df),
            'data_quality_neural': self._assess_data_quality_neural(df)
        }
        
        return results
    
    def _process_text_classification(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Processa classificação de texto usando redes neurais"""
        text_results = {}
        
        # Identifica colunas de texto
        text_columns = df.select_dtypes(include=['object']).columns
        
        for column in text_columns:
            text_data = df[column].dropna().astype(str)
            
            if len(text_data) == 0:
                continue
            
            # Limita a amostra para performance
            sample_size = min(1000, len(text_data))
            text_sample = text_data.sample(n=sample_size, random_state=42)
            
            try:
                # Classificação automática de categorias
                categories = self.text_classifier.classify_text_categories(text_sample.tolist())
                
                # Análise de sentimentos (se aplicável)
                sentiments = self.text_classifier.analyze_sentiment(text_sample.tolist())
                
                # Extração de entidades
                entities = self.text_classifier.extract_entities(text_sample.tolist())
                
                text_results[column] = {
                    'categories': categories,
                    'sentiments': sentiments,
                    'entities': entities,
                    'sample_size': sample_size
                }
                
            except Exception as e:
                logger.warning(f"Erro na classificação de texto da coluna {column}: {e}")
                text_results[column] = {'error': str(e)}
        
        return text_results
    
    def _process_anomaly_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detecção de anomalias usando redes neurais"""
        anomaly_results = {}
        
        # Anomalias em dados numéricos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_data = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            try:
                neural_anomalies = self.anomaly_detector.detect_anomalies(
                    numeric_data.values
                )
                
                anomaly_results['numeric_anomalies'] = {
                    'anomaly_indices': neural_anomalies['anomaly_indices'],
                    'anomaly_scores': neural_anomalies['anomaly_scores'],
                    'threshold': neural_anomalies['threshold'],
                    'total_anomalies': len(neural_anomalies['anomaly_indices'])
                }
                
            except Exception as e:
                logger.warning(f"Erro na detecção de anomalias numéricas: {e}")
                anomaly_results['numeric_anomalies'] = {'error': str(e)}
        
        # Anomalias em dados textuais
        text_cols = df.select_dtypes(include=['object']).columns
        for column in text_cols[:3]:  # Limita a 3 colunas para performance
            text_data = df[column].dropna().astype(str)
            
            if len(text_data) < 10:
                continue
            
            try:
                text_anomalies = self.anomaly_detector.detect_text_anomalies(
                    text_data.tolist()
                )
                
                anomaly_results[f'{column}_text_anomalies'] = text_anomalies
                
            except Exception as e:
                logger.warning(f"Erro na detecção de anomalias textuais da coluna {column}: {e}")
                anomaly_results[f'{column}_text_anomalies'] = {'error': str(e)}
        
        return anomaly_results
    
    def _process_similarity_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de similaridade usando embeddings neurais"""
        similarity_results = {}
        
        # Similaridade entre colunas de texto
        text_cols = df.select_dtypes(include=['object']).columns
        
        if len(text_cols) >= 2:
            try:
                # Calcula similaridade semântica entre colunas
                column_similarities = self.similarity_engine.calculate_column_similarities(
                    df, text_cols.tolist()
                )
                
                similarity_results['column_similarities'] = column_similarities
                
                # Encontra duplicatas semânticas
                for column in text_cols[:3]:  # Limita para performance
                    semantic_duplicates = self.similarity_engine.find_semantic_duplicates(
                        df[column].dropna().astype(str).tolist()
                    )
                    
                    similarity_results[f'{column}_semantic_duplicates'] = semantic_duplicates
                
            except Exception as e:
                logger.warning(f"Erro na análise de similaridade: {e}")
                similarity_results['error'] = str(e)
        
        return similarity_results
    
    def _process_pattern_recognition(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Reconhecimento de padrões usando redes neurais"""
        pattern_results = {}
        
        try:
            # Padrões em sequências numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for column in numeric_cols[:5]:  # Limita para performance
                numeric_data = df[column].dropna().values
                
                if len(numeric_data) >= 10:
                    patterns = self.text_classifier.detect_numeric_patterns(numeric_data)
                    pattern_results[f'{column}_patterns'] = patterns
            
            # Padrões em sequências textuais
            text_cols = df.select_dtypes(include=['object']).columns
            for column in text_cols[:3]:  # Limita para performance
                text_data = df[column].dropna().astype(str).tolist()
                
                if len(text_data) >= 10:
                    text_patterns = self.text_classifier.detect_text_patterns(text_data)
                    pattern_results[f'{column}_text_patterns'] = text_patterns
            
        except Exception as e:
            logger.warning(f"Erro no reconhecimento de padrões: {e}")
            pattern_results['error'] = str(e)
        
        return pattern_results
    
    def _assess_data_quality_neural(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Avalia qualidade dos dados usando abordagens neurais"""
        quality_results = {}
        
        try:
            # Score de qualidade baseado em embeddings
            text_cols = df.select_dtypes(include=['object']).columns
            
            if len(text_cols) > 0:
                quality_scores = []
                
                for column in text_cols[:5]:  # Limita para performance
                    text_data = df[column].dropna().astype(str)
                    
                    if len(text_data) > 0:
                        # Usa embeddings para avaliar qualidade semântica
                        quality_score = self._calculate_semantic_quality_score(text_data.tolist())
                        quality_scores.append(quality_score)
                
                if quality_scores:
                    quality_results['semantic_quality'] = {
                        'overall_score': np.mean(quality_scores),
                        'column_scores': quality_scores,
                        'quality_level': self._classify_quality_level(np.mean(quality_scores))
                    }
            
            # Consistência neural
            consistency_score = self._calculate_neural_consistency(df)
            quality_results['neural_consistency'] = consistency_score
            
        except Exception as e:
            logger.warning(f"Erro na avaliação de qualidade neural: {e}")
            quality_results['error'] = str(e)
        
        return quality_results
    
    def _calculate_semantic_quality_score(self, texts: List[str]) -> float:
        """Calcula score de qualidade semântica"""
        try:
            # Limita amostra para performance
            sample_texts = texts[:100] if len(texts) > 100 else texts
            
            # Gera embeddings
            embeddings = self.embedding_engine.generate_embeddings_batch(sample_texts)
            embeddings_array = np.array(embeddings)
            
            # Calcula diversidade semântica
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Remove diagonal (auto-similaridade)
            mask = np.ones_like(similarity_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            similarities = similarity_matrix[mask]
            
            # Qualidade baseada na diversidade (menos similaridade = mais qualidade)
            avg_similarity = np.mean(similarities)
            quality_score = 1 - min(avg_similarity, 1.0)
            
            return float(quality_score)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de qualidade semântica: {e}")
            return 0.5  # Score neutro em caso de erro
    
    def _calculate_neural_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula consistência dos dados usando abordagens neurais"""
        try:
            consistency_scores = []
            
            # Verifica consistência em colunas categóricas
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for column in categorical_cols[:5]:  # Limita para performance
                values = df[column].dropna().astype(str).tolist()
                
                if len(values) > 10:
                    # Usa embeddings para detectar inconsistências semânticas
                    embeddings = self.embedding_engine.generate_embeddings_batch(values[:50])
                    
                    # Calcula variabilidade semântica
                    embeddings_array = np.array(embeddings)
                    centroid = np.mean(embeddings_array, axis=0)
                    distances = np.linalg.norm(embeddings_array - centroid, axis=1)
                    
                    # Consistência baseada na proximidade ao centroide
                    consistency = 1 - (np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0)
                    consistency_scores.append(max(0, min(1, consistency)))
            
            if consistency_scores:
                return {
                    'overall_consistency': float(np.mean(consistency_scores)),
                    'column_consistencies': consistency_scores,
                    'consistency_level': self._classify_quality_level(np.mean(consistency_scores))
                }
            else:
                return {'overall_consistency': 0.5, 'message': 'Dados insuficientes para análise'}
                
        except Exception as e:
            logger.warning(f"Erro no cálculo de consistência neural: {e}")
            return {'overall_consistency': 0.5, 'error': str(e)}
    
    def _classify_quality_level(self, score: float) -> str:
        """Classifica nível de qualidade baseado no score"""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def generate_neural_insights(self, df: pd.DataFrame, 
                               neural_results: Dict[str, Any]) -> List[str]:
        """Gera insights baseados na análise neural"""
        insights = []
        
        try:
            # Insights sobre classificação de texto
            if 'text_classification' in neural_results:
                text_class = neural_results['text_classification']
                for column, results in text_class.items():
                    if 'categories' in results:
                        insights.append(f"Coluna '{column}': {len(results['categories'])} categorias identificadas")
            
            # Insights sobre anomalias
            if 'anomaly_detection' in neural_results:
                anomaly_data = neural_results['anomaly_detection']
                if 'numeric_anomalies' in anomaly_data:
                    total_anomalies = anomaly_data['numeric_anomalies'].get('total_anomalies', 0)
                    if total_anomalies > 0:
                        insights.append(f"{total_anomalies} anomalias numéricas detectadas pela rede neural")
            
            # Insights sobre qualidade
            if 'data_quality_neural' in neural_results:
                quality_data = neural_results['data_quality_neural']
                if 'semantic_quality' in quality_data:
                    quality_level = quality_data['semantic_quality'].get('quality_level', 'unknown')
                    insights.append(f"Qualidade semântica dos dados: {quality_level}")
            
            if not insights:
                insights.append("Análise neural concluída - dados processados com sucesso")
            
        except Exception as e:
            logger.warning(f"Erro na geração de insights neurais: {e}")
            insights.append("Análise neural concluída com limitações")
        
        return insights

# Instância global que será configurada se necessário
neural_processor = None

def get_neural_processor(embedding_engine: EmbeddingEngine = None):
    """Obtém o processador neural, inicializando se necessário"""
    global neural_processor
    if neural_processor is None:
        engine = embedding_engine or get_embedding_engine()
        neural_processor = NeuralProcessor(embedding_engine=engine)
    return neural_processor
