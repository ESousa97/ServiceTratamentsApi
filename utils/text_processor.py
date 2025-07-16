import re
import unicodedata
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Processador avançado de texto com múltiplas funcionalidades"""
    
    def __init__(self):
        self.stop_words_pt = {
            'a', 'o', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não',
            'que', 'se', 'na', 'por', 'mais', 'as', 'os', 'no', 'ao', 'dos', 'das', 'como',
            'mas', 'foi', 'ele', 'ela', 'entre', 'era', 'ser', 'ter', 'seu', 'sua', 'ou',
            'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo',
            'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'ter',
            'seus', 'suas', 'num', 'numa', 'pelos', 'pelas', 'esse', 'essa', 'esses',
            'essas', 'aquele', 'aquela', 'aqueles', 'aquelas'
        }
        
        self.stop_words_en = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will',
            'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'were', 'been',
            'their', 'said', 'each', 'which', 'do', 'how', 'if', 'who', 'oil', 'sit'
        }
    
    def clean_text(self, text: str, remove_accents: bool = True, 
                   to_lowercase: bool = True, remove_punctuation: bool = True) -> str:
        """Limpa e normaliza texto"""
        if not text or not isinstance(text, str):
            return ""
        
        cleaned = text.strip()
        
        # Remove acentos
        if remove_accents:
            cleaned = self.remove_accents(cleaned)
        
        # Converte para minúsculas
        if to_lowercase:
            cleaned = cleaned.lower()
        
        # Remove pontuação
        if remove_punctuation:
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        
        # Remove espaços múltiplos
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def remove_accents(self, text: str) -> str:
        """Remove acentos do texto"""
        if not text:
            return ""
        
        # Normaliza unicode e remove acentos
        normalized = unicodedata.normalize('NFKD', text)
        ascii_text = normalized.encode('ASCII', 'ignore').decode('utf-8')
        
        return ascii_text
    
    def remove_stop_words(self, text: str, language: str = 'pt') -> str:
        """Remove stop words do texto"""
        if not text:
            return ""
        
        stop_words = self.stop_words_pt if language == 'pt' else self.stop_words_en
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        return ' '.join(filtered_words)
    
    def extract_keywords(self, text: str, min_length: int = 3, 
                        max_keywords: int = 10) -> List[str]:
        """Extrai palavras-chave do texto"""
        if not text:
            return []
        
        # Limpa o texto
        cleaned = self.clean_text(text)
        
        # Remove stop words
        without_stop_words = self.remove_stop_words(cleaned)
        
        # Extrai palavras
        words = without_stop_words.split()
        
        # Filtra por comprimento
        keywords = [word for word in words if len(word) >= min_length]
        
        # Conta frequências
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Ordena por frequência
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Retorna top keywords
        return [word for word, freq in sorted_keywords[:max_keywords]]
    
    def detect_language(self, text: str) -> str:
        """Detecta idioma do texto (português/inglês)"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Conta palavras indicativas
        pt_count = len(words.intersection(self.stop_words_pt))
        en_count = len(words.intersection(self.stop_words_en))
        
        if pt_count > en_count:
            return 'portuguese'
        elif en_count > pt_count:
            return 'english'
        else:
            return 'unknown'
    
    def extract_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extrai padrões comuns do texto"""
        if not text:
            return {}
        
        patterns = {
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            'phones': re.findall(r'\b(?:\+?55\s?)?(?:\(\d{2}\)|\d{2})\s?\d{4,5}-?\d{4}\b', text),
            'cpf': re.findall(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b', text),
            'cnpj': re.findall(r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b', text),
            'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
            'urls': re.findall(r'https?://[^\s]+', text),
            'currency': re.findall(r'R\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?', text),
            'cep': re.findall(r'\b\d{5}-?\d{3}\b', text)
        }
        
        # Remove padrões vazios
        return {k: v for k, v in patterns.items() if v}
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calcula métricas de legibilidade do texto"""
        if not text:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}
        
        # Divide em sentenças
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Divide em palavras
        words = text.split()
        
        if not sentences or not words:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}
        
        # Métricas básicas
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Estimativa simplificada do Flesch Score (adaptada para português)
        flesch_score = 248.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        flesch_score = max(0, min(100, flesch_score))  # Limita entre 0-100
        
        return {
            'flesch_score': round(flesch_score, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'total_sentences': len(sentences),
            'total_words': len(words)
        }
    
    def similarity_levenshtein(self, text1: str, text2: str) -> float:
        """Calcula similaridade usando distância de Levenshtein"""
        if not text1 or not text2:
            return 0.0
        
        # Implementação simples da distância de Levenshtein
        len1, len2 = len(text1), len(text2)
        
        if len1 == 0:
            return 0.0 if len2 == 0 else 0.0
        if len2 == 0:
            return 0.0
        
        # Matriz de distâncias
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Inicializa primeira linha e coluna
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j
        
        # Calcula distâncias
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if text1[i-1] == text2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # Deletion
                    matrix[i][j-1] + 1,      # Insertion
                    matrix[i-1][j-1] + cost  # Substitution
                )
        
        # Converte distância em similaridade (0-1)
        max_len = max(len1, len2)
        distance = matrix[len1][len2]
        similarity = 1 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def correct_common_errors(self, text: str) -> str:
        """Corrige erros ortográficos comuns"""
        if not text:
            return ""
        
        # Dicionário de correções comuns
        corrections = {
            # Português
            'voce': 'você', 'nao': 'não', 'sao': 'são', 'entao': 'então',
            'tambem': 'também', 'porem': 'porém', 'alem': 'além',
            'atraves': 'através', 'pais': 'país', 'facil': 'fácil',
            'util': 'útil', 'proximo': 'próximo', 'ultimo': 'último',
            'numero': 'número', 'minimo': 'mínimo', 'maximo': 'máximo',
            'otimo': 'ótimo', 'pessimo': 'péssimo', 'rapido': 'rápido',
            'publico': 'público', 'basico': 'básico', 'economico': 'econômico',
            'tecnico': 'técnico', 'matematica': 'matemática',
            'informatica': 'informática', 'solucao': 'solução',
            'informacao': 'informação', 'operacao': 'operação',
            'configuracao': 'configuração', 'aplicacao': 'aplicação',
            
            # Inglês
            'recieve': 'receive', 'seperate': 'separate', 'occured': 'occurred',
            'begining': 'beginning', 'existance': 'existence', 'maintainance': 'maintenance',
            'accross': 'across', 'sucessful': 'successful', 'enviroment': 'environment'
        }
        
        corrected = text
        for wrong, correct in corrections.items():
            # Substitui palavra completa (case insensitive)
            pattern = r'\b' + re.escape(wrong) + r'\b'
            corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def tokenize(self, text: str, preserve_case: bool = False) -> List[str]:
        """Tokeniza texto em palavras"""
        if not text:
            return []
        
        # Remove pontuação e divide em palavras
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        
        if not preserve_case:
            cleaned = cleaned.lower()
        
        tokens = cleaned.split()
        
        # Remove tokens vazios
        return [token for token in tokens if token.strip()]
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extrai n-gramas do texto"""
        if not text or n < 1:
            return []
        
        tokens = self.tokenize(text)
        
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calcula estatísticas completas do texto"""
        if not text:
            return {}
        
        # Estatísticas básicas
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text)) - 1  # -1 porque split cria item vazio no final
        
        # Estatísticas avançadas
        unique_words = len(set(text.lower().split()))
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        # Densidade lexical
        lexical_density = unique_words / word_count if word_count > 0 else 0
        
        # Padrões encontrados
        patterns = self.extract_patterns(text)
        pattern_count = sum(len(pattern_list) for pattern_list in patterns.values())
        
        # Legibilidade
        readability = self.calculate_readability(text)
        
        return {
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'word_count': word_count,
            'sentence_count': max(1, sentence_count),  # Pelo menos 1 sentença
            'unique_words': unique_words,
            'lexical_density': round(lexical_density, 3),
            'avg_word_length': round(avg_word_length, 2),
            'patterns_found': pattern_count,
            'readability': readability,
            'language': self.detect_language(text)
        }

# Instância global
text_processor = TextProcessor()
