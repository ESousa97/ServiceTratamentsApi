import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import difflib
import re
import logging
from models.embeddings import EmbeddingEngine, TextPreprocessor
from core.memory_manager import memory_manager
from config.settings import config

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

class SemanticEngine:
    """Engine para similaridade semântica e correção ortográfica"""
    
    def __init__(self, similarity_threshold: float = None, embedding_engine: EmbeddingEngine = None):
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        self.text_preprocessor = TextPreprocessor()
        self.embedding_engine = embedding_engine or get_embedding_engine()
        
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Calcula matriz de similaridade semântica"""
        if not texts:
            return np.array([])
        
        # Limpa textos
        cleaned_texts = [self.text_preprocessor.clean_text(text) for text in texts]
        
        # Gera embeddings
        embeddings = self.embedding_engine.generate_embeddings_batch(cleaned_texts)
        embeddings_array = np.array(embeddings)
        
        # Calcula similaridade de cosseno
        similarity_matrix = cosine_similarity(embeddings_array)
        
        return similarity_matrix
    
    def find_similar_texts(self, target_text: str, candidate_texts: List[str], 
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """Encontra textos mais similares ao texto alvo"""
        if not candidate_texts:
            return []
        
        # Prepara textos
        all_texts = [target_text] + candidate_texts
        
        # Calcula similaridades
        similarity_matrix = self.calculate_similarity_matrix(all_texts)
        
        # Extrai similaridades com o texto alvo (primeira linha/coluna)
        similarities = similarity_matrix[0, 1:]  # Exclui auto-similaridade
        
        # Ordena por similaridade
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i in sorted_indices[:top_k]:
            similarity_score = similarities[i]
            if similarity_score >= self.similarity_threshold:
                results.append((candidate_texts[i], similarity_score))
        
        return results
    
    def group_similar_texts(self, texts: List[str], 
                           eps: float = None, min_samples: int = 2) -> Dict[int, List[str]]:
        """Agrupa textos similares usando clustering"""
        if not texts or len(texts) < 2:
            return {0: texts}
        
        eps = eps or (1 - self.similarity_threshold)
        
        # Gera embeddings
        embeddings = self.embedding_engine.generate_embeddings_batch(texts)
        embeddings_array = np.array(embeddings)
        
        # Clustering com DBSCAN
        clustering = DBSCAN(
            eps=eps, 
            min_samples=min_samples, 
            metric='cosine'
        ).fit(embeddings_array)
        
        # Agrupa por labels
        groups = {}
        for i, label in enumerate(clustering.labels_):
            if label not in groups:
                groups[label] = []
            groups[label].append(texts[i])
        
        return groups
    
    def correct_spelling_contextual(self, text: str, 
                                  context_texts: List[str] = None) -> str:
        """Corrige ortografia baseado no contexto"""
        if not text or not text.strip():
            return text
        
        # Tokeniza palavras
        words = re.findall(r'\b\w+\b', text.lower())
        corrected_words = []
        
        for word in words:
            corrected_word = self._correct_word(word, context_texts)
            corrected_words.append(corrected_word)
        
        # Reconstrói texto mantendo pontuação
        corrected_text = text
        for original, corrected in zip(words, corrected_words):
            if original != corrected:
                # Substitui preservando case
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                corrected_text = pattern.sub(corrected, corrected_text, count=1)
        
        return corrected_text
    
    def _correct_word(self, word: str, context_texts: List[str] = None) -> str:
        """Corrige palavra individual"""
        if len(word) < 3:  # Palavras muito curtas não são corrigidas
            return word
        
        # Busca em contexto se fornecido
        if context_texts:
            context_words = set()
            for text in context_texts:
                context_words.update(re.findall(r'\b\w+\b', text.lower()))
            
            # Encontra palavra mais similar no contexto
            close_matches = difflib.get_close_matches(
                word, context_words, n=1, cutoff=0.8
            )
            
            if close_matches:
                return close_matches[0]
        
        # Fallback: correções comuns
        common_corrections = self._get_common_corrections()
        if word in common_corrections:
            return common_corrections[word]
        
        return word
    
    def _get_common_corrections(self) -> Dict[str, str]:
        """Retorna correções ortográficas comuns"""
        return {
            # Correções específicas para português
            'voce': 'você',
            'nao': 'não',
            'sao': 'são',
            'entao': 'então',
            'tambem': 'também',
            'porem': 'porém',
            'alem': 'além',
            'atraves': 'através',
            'pais': 'país',
            'facil': 'fácil',
            'util': 'útil',
            'proximo': 'próximo',
            'ultimo': 'último',
            'numero': 'número',
            'minimo': 'mínimo',
            'maximo': 'máximo',
            'otimo': 'ótimo',
            'pessimo': 'péssimo',
            'rapido': 'rápido',
            'publico': 'público',
            'basico': 'básico',
            'economico': 'econômico',
            'tecnico': 'técnico',
            'matematica': 'matemática',
            'informatica': 'informática',
            'problemas': 'problemas',
            'solucao': 'solução',
            'informacao': 'informação',
            'operacao': 'operação',
            'configuracao': 'configuração',
            'aplicacao': 'aplicação',
        }
    
    def detect_language_inconsistencies(self, texts: List[str]) -> Dict[str, List[str]]:
        """Detecta inconsistências de idioma nos textos"""
        language_groups = {
            'portuguese': [],
            'english': [],
            'mixed': [],
            'unknown': []
        }
        
        for text in texts:
            lang = self._detect_text_language(text)
            if lang in language_groups:
                language_groups[lang].append(text)
            else:
                language_groups['unknown'].append(text)
        
        return language_groups
    
    def _detect_text_language(self, text: str) -> str:
        """Detecta idioma do texto (simplificado)"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        # Palavras indicativas de português
        portuguese_words = [
            'não', 'são', 'você', 'também', 'então', 'através', 'informação',
            'solução', 'configuração', 'aplicação', 'operação', 'português',
            'número', 'mínimo', 'máximo', 'público', 'técnico', 'econômico'
        ]
        
        # Palavras indicativas de inglês
        english_words = [
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'they',
            'have', 'been', 'their', 'said', 'each', 'which', 'there',
            'what', 'would', 'about', 'into', 'time', 'very', 'when'
        ]
        
        portuguese_count = sum(1 for word in portuguese_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if portuguese_count > english_count and portuguese_count > 0:
            return 'portuguese'
        elif english_count > portuguese_count and english_count > 0:
            return 'english'
        elif portuguese_count > 0 and english_count > 0:
            return 'mixed'
        else:
            return 'unknown'

class SemanticAnalyzer:
    """Analisador semântico avançado para datasets"""
    
    def __init__(self, embedding_engine: EmbeddingEngine = None):
        self.semantic_engine = SemanticEngine(embedding_engine=embedding_engine)
    
    @memory_manager.memory_limiter
    def analyze_column_semantics(self, df: pd.DataFrame, 
                                column: str) -> Dict[str, any]:
        """Analisa semântica de uma coluna"""
        if column not in df.columns:
            raise ValueError(f"Coluna '{column}' não encontrada")
        
        # Extrai textos únicos não-nulos
        texts = df[column].dropna().astype(str).unique().tolist()
        
        if not texts:
            return {'error': 'Coluna não contém dados válidos'}
        
        # Análise de similaridade
        similarity_analysis = self._analyze_text_similarity(texts)
        
        # Análise de agrupamentos
        groups = self.semantic_engine.group_similar_texts(texts)
        
        # Análise de qualidade do texto
        quality_analysis = self._analyze_text_quality(texts)
        
        # Detecção de idiomas
        language_analysis = self.semantic_engine.detect_language_inconsistencies(texts)
        
        return {
            'column': column,
            'total_unique_values': len(texts),
            'similarity_analysis': similarity_analysis,
            'semantic_groups': {
                str(k): v for k, v in groups.items()
            },
            'text_quality': quality_analysis,
            'language_distribution': {
                k: len(v) for k, v in language_analysis.items()
            },
            'recommendations': self._generate_recommendations(
                similarity_analysis, groups, quality_analysis, language_analysis
            )
        }
    
    def _analyze_text_similarity(self, texts: List[str]) -> Dict[str, any]:
        """Analisa distribuição de similaridades"""
        if len(texts) < 2:
            return {'avg_similarity': 0, 'max_similarity': 0, 'min_similarity': 0}
        
        # Limita análise para economia de memória
        sample_size = min(100, len(texts))
        sample_texts = texts[:sample_size]
        
        similarity_matrix = self.semantic_engine.calculate_similarity_matrix(sample_texts)
        
        # Remove diagonal (auto-similaridade)
        mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        similarities = similarity_matrix[mask]
        
        return {
            'avg_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'std_similarity': float(np.std(similarities)),
            'high_similarity_pairs': int(np.sum(similarities > 0.8)),
            'sample_size': sample_size
        }
    
    def _analyze_text_quality(self, texts: List[str]) -> Dict[str, any]:
        """Analisa qualidade dos textos"""
        if not texts:
            return {}
        
        # Estatísticas básicas
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Problemas de qualidade
        empty_texts = sum(1 for text in texts if not text.strip())
        numeric_only = sum(1 for text in texts if text.strip().isdigit())
        single_char = sum(1 for text in texts if len(text.strip()) == 1)
        
        # Caracteres especiais excessivos
        special_char_heavy = sum(
            1 for text in texts 
            if len(re.findall(r'[^\w\s]', text)) > len(text) * 0.3
        )
        
        return {
            'avg_length': np.mean(lengths),
            'avg_word_count': np.mean(word_counts),
            'empty_texts': empty_texts,
            'numeric_only': numeric_only,
            'single_character': single_char,
            'special_char_heavy': special_char_heavy,
            'quality_score': self._calculate_quality_score(
                empty_texts, numeric_only, single_char, special_char_heavy, len(texts)
            )
        }
    
    def _calculate_quality_score(self, empty: int, numeric: int, 
                               single: int, special: int, total: int) -> float:
        """Calcula score de qualidade (0-1)"""
        if total == 0:
            return 0
        
        problems = empty + numeric + single + special
        return max(0, 1 - (problems / total))
    
    def _generate_recommendations(self, similarity_analysis: Dict, 
                                groups: Dict, quality_analysis: Dict,
                                language_analysis: Dict) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        # Recomendações de similaridade
        if similarity_analysis.get('avg_similarity', 0) > 0.8:
            recommendations.append("Muitos valores similares detectados - considere agrupamento ou normalização")
        
        # Recomendações de agrupamento
        noise_group = groups.get(-1, [])
        if len(noise_group) > len(groups) * 0.3:
            recommendations.append("Muitos valores únicos sem agrupamento - dados podem estar muito fragmentados")
        
        # Recomendações de qualidade
        quality_score = quality_analysis.get('quality_score', 1)
        if quality_score < 0.7:
            recommendations.append("Qualidade dos dados baixa - considere limpeza e padronização")
        
        if quality_analysis.get('empty_texts', 0) > 0:
            recommendations.append("Valores vazios detectados - considere tratamento de dados faltantes")
        
        # Recomendações de idioma
        mixed_count = len(language_analysis.get('mixed', []))
        total_with_lang = sum(len(v) for k, v in language_analysis.items() if k != 'unknown')
        
        if mixed_count > total_with_lang * 0.1:
            recommendations.append("Textos com idiomas misturados detectados - considere separação por idioma")
        
        if not recommendations:
            recommendations.append("Dados parecem estar em boa qualidade para análise")
        
        return recommendations

class SemanticDataProcessor:
    """Processador de dados com capacidades semânticas"""
    
    def __init__(self, embedding_engine: EmbeddingEngine = None):
        self.semantic_engine = SemanticEngine(embedding_engine=embedding_engine)
        self.analyzer = SemanticAnalyzer(embedding_engine=embedding_engine)
    
    def process_dataframe_semantics(self, df: pd.DataFrame, 
                                  text_columns: List[str] = None) -> Dict[str, any]:
        """Processa semântica completa do DataFrame"""
        if text_columns is None:
            # Auto-detecta colunas de texto
            text_columns = self._detect_text_columns(df)
        
        results = {
            'text_columns': text_columns,
            'column_analyses': {},
            'cross_column_analysis': {},
            'overall_recommendations': []
        }
        
        # Analisa cada coluna individualmente
        for column in text_columns:
            try:
                analysis = self.analyzer.analyze_column_semantics(df, column)
                results['column_analyses'][column] = analysis
            except Exception as e:
                logger.error(f"Erro na análise da coluna {column}: {e}")
                results['column_analyses'][column] = {'error': str(e)}
        
        # Análise entre colunas
        if len(text_columns) > 1:
            results['cross_column_analysis'] = self._analyze_cross_columns(
                df, text_columns
            )
        
        # Recomendações gerais
        results['overall_recommendations'] = self._generate_overall_recommendations(
            results['column_analyses'], results['cross_column_analysis']
        )
        
        return results
    
    def _detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Detecta automaticamente colunas de texto"""
        text_columns = []
        
        for column in df.columns:
            sample = df[column].dropna().head(10)
            
            if len(sample) == 0:
                continue
            
            # Verifica se contém strings
            string_count = sum(1 for val in sample if isinstance(val, str))
            
            # Verifica se strings contêm palavras (não só números)
            word_count = 0
            for val in sample:
                if isinstance(val, str) and len(val.split()) > 1:
                    word_count += 1
            
            # É coluna de texto se maioria são strings com palavras
            if string_count > len(sample) * 0.7 and word_count > 0:
                text_columns.append(column)
        
        return text_columns
    
    def _analyze_cross_columns(self, df: pd.DataFrame, 
                             text_columns: List[str]) -> Dict[str, any]:
        """Analisa relações entre colunas de texto"""
        cross_analysis = {
            'column_similarities': {},
            'shared_vocabulary': {},
            'correlation_insights': []
        }
        
        # Análise de similaridade entre colunas
        for i, col1 in enumerate(text_columns):
            for col2 in text_columns[i+1:]:
                similarity = self._calculate_column_similarity(df, col1, col2)
                cross_analysis['column_similarities'][f"{col1}_vs_{col2}"] = similarity
        
        return cross_analysis
    
    def _calculate_column_similarity(self, df: pd.DataFrame, 
                                   col1: str, col2: str) -> float:
        """Calcula similaridade semântica entre duas colunas"""
        # Extrai amostras de ambas as colunas
        sample1 = df[col1].dropna().astype(str).head(50).tolist()
        sample2 = df[col2].dropna().astype(str).head(50).tolist()
        
        if not sample1 or not sample2:
            return 0.0
        
        # Gera embeddings médios para cada coluna
        embeddings1 = self.semantic_engine.embedding_engine.generate_embeddings_batch(sample1)
        embeddings2 = self.semantic_engine.embedding_engine.generate_embeddings_batch(sample2)
        
        avg_embedding1 = np.mean(embeddings1, axis=0)
        avg_embedding2 = np.mean(embeddings2, axis=0)
        
        # Calcula similaridade de cosseno
        similarity = cosine_similarity([avg_embedding1], [avg_embedding2])[0][0]
        
        return float(similarity)
    
    def _generate_overall_recommendations(self, column_analyses: Dict, 
                                        cross_analysis: Dict) -> List[str]:
        """Gera recomendações gerais para o dataset"""
        recommendations = []
        
        # Analisa qualidade geral
        quality_scores = []
        for analysis in column_analyses.values():
            if 'text_quality' in analysis:
                quality_scores.append(analysis['text_quality'].get('quality_score', 1))
        
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            if avg_quality < 0.6:
                recommendations.append("Qualidade geral dos dados textuais é baixa - priorize limpeza")
            elif avg_quality > 0.9:
                recommendations.append("Excelente qualidade dos dados textuais")
        
        # Analisa similaridades entre colunas
        similarities = cross_analysis.get('column_similarities', {})
        high_similarities = [sim for sim in similarities.values() if sim > 0.8]
        
        if len(high_similarities) > 0:
            recommendations.append("Colunas altamente similares detectadas - considere consolidação")
        
        return recommendations

# Instância global que será configurada pelo main.py
semantic_engine = None
semantic_analyzer = None
semantic_processor = None

def initialize_semantic_components(embedding_engine: EmbeddingEngine = None):
    """Inicializa componentes semânticos com a engine de embeddings"""
    global semantic_engine, semantic_analyzer, semantic_processor
    
    engine = embedding_engine or get_embedding_engine()
    semantic_engine = SemanticEngine(embedding_engine=engine)
    semantic_analyzer = SemanticAnalyzer(embedding_engine=engine)
    semantic_processor = SemanticDataProcessor(embedding_engine=engine)

def get_semantic_processor():
    """Obtém o processador semântico, inicializando se necessário"""
    global semantic_processor
    if semantic_processor is None:
        initialize_semantic_components()
    return semantic_processor

# Para compatibilidade com imports existentes
def get_semantic_engine():
    """Obtém a engine semântica, inicializando se necessário"""
    global semantic_engine
    if semantic_engine is None:
        initialize_semantic_components()
    return semantic_engine

def get_semantic_analyzer():
    """Obtém o analisador semântico, inicializando se necessário"""
    global semantic_analyzer
    if semantic_analyzer is None:
        initialize_semantic_components()
    return semantic_analyzer
