import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import difflib
import re
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Pré-processador de texto para embeddings"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Limpa e normaliza texto"""
        if not text:
            return ""
        
        # Remove caracteres especiais excessivos
        text = re.sub(r'\s+', ' ', text)  # Múltiplos espaços
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)  # Mantém pontuação básica
        
        # Normaliza
        text = text.strip().lower()
        
        return text

class SemanticEngine:
    """Engine para similaridade semântica e correção ortográfica"""
    
    def __init__(self, similarity_threshold: float = 0.7, embedding_engine=None):
        self.similarity_threshold = similarity_threshold
        self.text_preprocessor = TextPreprocessor()
        self.embedding_engine = embedding_engine
        
    def get_embedding_engine(self):
        """Obtém embedding engine com lazy loading"""
        if self.embedding_engine is None:
            try:
                # Import dinâmico para evitar circular
                import importlib
                embeddings_module = importlib.import_module('models.embeddings')
                EmbeddingEngine = getattr(embeddings_module, 'EmbeddingEngine')
                self.embedding_engine = EmbeddingEngine()
            except Exception as e:
                logger.warning(f"Erro ao carregar embedding engine: {e}")
                # Retorna None - funções irão usar fallbacks
                return None
        return self.embedding_engine
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Calcula matriz de similaridade semântica"""
        if not texts:
            return np.array([])
        
        # Limpa textos
        cleaned_texts = [self.text_preprocessor.clean_text(text) for text in texts]
        
        # Tenta usar embeddings se disponível
        engine = self.get_embedding_engine()
        if engine:
            try:
                embeddings = engine.generate_embeddings_batch(cleaned_texts)
                embeddings_array = np.array(embeddings)
                
                # Calcula similaridade de cosseno
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(embeddings_array)
                return similarity_matrix
            except Exception as e:
                logger.warning(f"Erro no cálculo de similaridade semântica: {e}")
        
        # Fallback para similaridade léxica simples
        return self._calculate_lexical_similarity_matrix(cleaned_texts)
    
    def _calculate_lexical_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Fallback: calcula similaridade léxica simples"""
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Usa similaridade de Jaccard baseada em palavras
                    words_i = set(texts[i].split())
                    words_j = set(texts[j].split())
                    
                    if len(words_i) == 0 and len(words_j) == 0:
                        similarity = 1.0
                    elif len(words_i) == 0 or len(words_j) == 0:
                        similarity = 0.0
                    else:
                        intersection = len(words_i.intersection(words_j))
                        union = len(words_i.union(words_j))
                        similarity = intersection / union if union > 0 else 0.0
                    
                    similarity_matrix[i][j] = similarity
        
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
        
        if similarity_matrix.size == 0:
            return []
        
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
        
        # Tenta usar clustering semântico
        engine = self.get_embedding_engine()
        if engine:
            try:
                embeddings = engine.generate_embeddings_batch(texts)
                embeddings_array = np.array(embeddings)
                
                from sklearn.cluster import DBSCAN
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
            except Exception as e:
                logger.warning(f"Erro no clustering semântico: {e}")
        
        # Fallback para agrupamento léxico simples
        return self._simple_text_grouping(texts)
    
    def _simple_text_grouping(self, texts: List[str]) -> Dict[int, List[str]]:
        """Fallback: agrupamento simples baseado em palavras comuns"""
        groups = {}
        group_id = 0
        processed = set()
        
        for i, text in enumerate(texts):
            if i in processed:
                continue
            
            # Cria novo grupo
            current_group = [text]
            processed.add(i)
            
            # Busca textos similares
            words_i = set(text.lower().split())
            
            for j, other_text in enumerate(texts[i+1:], i+1):
                if j in processed:
                    continue
                
                words_j = set(other_text.lower().split())
                
                # Calcula similaridade Jaccard
                if len(words_i) > 0 and len(words_j) > 0:
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    similarity = intersection / union
                    
                    if similarity >= 0.3:  # Threshold para agrupamento
                        current_group.append(other_text)
                        processed.add(j)
            
            groups[group_id] = current_group
            group_id += 1
        
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

class SemanticDataProcessor:
    """Processador de dados com capacidades semânticas"""
    
    def __init__(self, embedding_engine=None):
        self.semantic_engine = SemanticEngine(embedding_engine=embedding_engine)
    
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
        
        # Análise simplificada para evitar dependências complexas
        for column in text_columns[:3]:  # Limita a 3 colunas para performance
            try:
                analysis = self._analyze_column_basic(df, column)
                results['column_analyses'][column] = analysis
            except Exception as e:
                logger.error(f"Erro na análise da coluna {column}: {e}")
                results['column_analyses'][column] = {'error': str(e)}
        
        # Recomendações gerais
        results['overall_recommendations'] = [
            "Análise semântica básica concluída",
            "Para análise avançada, verifique se os modelos de IA estão disponíveis"
        ]
        
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
    
    def _analyze_column_basic(self, df: pd.DataFrame, column: str) -> Dict[str, any]:
        """Análise básica de uma coluna"""
        if column not in df.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        # Extrai textos únicos não-nulos
        texts = df[column].dropna().astype(str).unique().tolist()
        
        if not texts:
            return {'error': 'Coluna não contém dados válidos'}
        
        # Análise básica
        analysis = {
            'column': column,
            'total_unique_values': len(texts),
            'avg_text_length': np.mean([len(text) for text in texts]),
            'language_distribution': self.semantic_engine.detect_language_inconsistencies(texts),
            'recommendations': ['Análise básica concluída']
        }
        
        return analysis

# Instâncias globais que serão configuradas
semantic_engine = None
semantic_processor = None

def initialize_semantic_components(embedding_engine=None):
    """Inicializa componentes semânticos com a engine de embeddings"""
    global semantic_engine, semantic_processor
    
    semantic_engine = SemanticEngine(embedding_engine=embedding_engine)
    semantic_processor = SemanticDataProcessor(embedding_engine=embedding_engine)

def get_semantic_processor():
    """Obtém o processador semântico, inicializando se necessário"""
    global semantic_processor
    if semantic_processor is None:
        initialize_semantic_components()
    return semantic_processor
