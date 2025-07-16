import numpy as np
import hashlib
import pickle
from typing import List, Dict, Union, Optional, Tuple
import logging

# Imports condicionais para evitar problemas circulares
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Engine para geração e cache de embeddings"""
    
    def __init__(self, model_name: str = None, device: str = None):
        # Configuração básica
        self.model_name = model_name or "paraphrase-MiniLM-L6-v2"
        
        if device:
            self.device = device
        elif TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.model = None
        self.tokenizer = None
        self.model_type = None
        
        # Configurações padrão
        self.max_sequence_length = 512
        
        # Inicializa modelo
        self._load_model()
        
        # Cache em memória simples
        self._memory_cache = {}
        
    def _load_model(self):
        """Carrega modelo de embeddings"""
        try:
            logger.info(f"Carregando modelo: {self.model_name}")
            
            # Tenta carregar com sentence-transformers primeiro
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.model = SentenceTransformer(self.model_name, device=self.device)
                    self.model_type = 'sentence_transformer'
                    logger.info("Modelo carregado com sentence-transformers")
                    return
                except Exception as e:
                    logger.warning(f"Falha no sentence-transformers: {e}")
            
            # Fallback para transformers
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.model.to(self.device)
                    self.model_type = 'transformer'
                    logger.info("Modelo carregado com transformers")
                    return
                except Exception as e:
                    logger.warning(f"Falha no transformers: {e}")
            
            # Fallback final - modelo dummy
            logger.warning("Usando modelo dummy para embeddings")
            self.model_type = 'dummy'
            self.model = None
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.model_type = 'dummy'
            self.model = None
    
    def _generate_text_hash(self, text: str) -> str:
        """Gera hash único para texto"""
        text_normalized = text.strip().lower()
        hash_input = f"{text_normalized}_{self.model_name}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Recupera embedding do cache em memória"""
        text_hash = self._generate_text_hash(text)
        return self._memory_cache.get(text_hash)
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Armazena embedding no cache em memória"""
        text_hash = self._generate_text_hash(text)
        
        # Limita tamanho do cache
        if len(self._memory_cache) > 1000:
            # Remove primeiro item (FIFO simples)
            first_key = next(iter(self._memory_cache))
            del self._memory_cache[first_key]
        
        self._memory_cache[text_hash] = embedding
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Gera embedding para texto único"""
        if not text or not text.strip():
            # Retorna embedding zero para texto vazio
            return np.zeros(self._get_embedding_dimension())
        
        # Verifica cache
        if use_cache:
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                return cached_embedding
        
        # Gera novo embedding
        try:
            if self.model_type == 'sentence_transformer':
                embedding = self.model.encode(text, convert_to_numpy=True)
            elif self.model_type == 'transformer':
                embedding = self._generate_with_transformer(text)
            else:
                # Modelo dummy - retorna vetor aleatório baseado no hash do texto
                embedding = self._generate_dummy_embedding(text)
            
            # Armazena no cache
            if use_cache:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            # Retorna embedding dummy em caso de erro
            return self._generate_dummy_embedding(text)
    
    def _generate_with_transformer(self, text: str) -> np.ndarray:
        """Gera embedding usando transformers direto"""
        try:
            # Tokeniza texto
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                max_length=self.max_sequence_length,
                truncation=True, 
                padding=True
            )
            
            # Move para device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Gera embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Usa pooling médio dos tokens
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Masked average pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * input_mask_expanded
            sum_embeddings = torch.sum(embeddings, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erro na geração com transformer: {e}")
            return self._generate_dummy_embedding(text)
    
    def _generate_dummy_embedding(self, text: str) -> np.ndarray:
        """Gera embedding dummy baseado no hash do texto"""
        # Usa hash do texto como seed para reprodutibilidade
        hash_value = hash(text) % (2**32)
        np.random.seed(hash_value)
        
        # Gera vetor aleatório normalizado
        embedding = np.random.randn(self._get_embedding_dimension())
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
    
    def generate_embeddings_batch(self, texts: List[str], 
                                 batch_size: int = None, 
                                 use_cache: bool = True) -> List[np.ndarray]:
        """Gera embeddings para lista de textos"""
        batch_size = batch_size or 32
        
        all_embeddings = []
        
        # Processa em batches para economizar memória
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Verifica cache para cada texto no batch
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                if use_cache:
                    cached = self._get_cached_embedding(text)
                    if cached is not None:
                        batch_embeddings.append((j, cached))
                        continue
                
                uncached_texts.append(text)
                uncached_indices.append(j)
            
            # Gera embeddings para textos não cacheados
            if uncached_texts:
                try:
                    if self.model_type == 'sentence_transformer' and self.model:
                        new_embeddings = self.model.encode(
                            uncached_texts, 
                            convert_to_numpy=True,
                            batch_size=min(batch_size, len(uncached_texts))
                        )
                    else:
                        # Processa individualmente
                        new_embeddings = [
                            self.generate_embedding(text, use_cache=False) 
                            for text in uncached_texts
                        ]
                    
                    # Adiciona novos embeddings aos resultados
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        batch_embeddings.append((idx, embedding))
                        
                        # Cache individual
                        if use_cache:
                            original_text = batch_texts[idx]
                            self._cache_embedding(original_text, embedding)
                    
                except Exception as e:
                    logger.error(f"Erro no batch {i//batch_size + 1}: {e}")
                    # Adiciona embeddings dummy para textos com erro
                    for idx in uncached_indices:
                        dummy_embedding = self._generate_dummy_embedding(batch_texts[idx])
                        batch_embeddings.append((idx, dummy_embedding))
            
            # Ordena embeddings pela posição original
            batch_embeddings.sort(key=lambda x: x[0])
            all_embeddings.extend([emb for _, emb in batch_embeddings])
            
            logger.debug(f"Processado batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
        
        return all_embeddings
    
    def _get_embedding_dimension(self) -> int:
        """Retorna dimensão dos embeddings"""
        if self.model_type == 'sentence_transformer' and self.model:
            return self.model.get_sentence_embedding_dimension()
        elif self.model_type == 'transformer':
            # Para transformers, faz uma inferência teste
            try:
                test_embedding = self._generate_with_transformer("test")
                return len(test_embedding)
            except:
                return 768  # Dimensão padrão para BERT-like models
        else:
            # Modelo dummy - dimensão fixa
            return 384  # Dimensão compatível com MiniLM
    
    def get_model_info(self) -> Dict[str, any]:
        """Retorna informações do modelo"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'device': self.device,
            'embedding_dimension': self._get_embedding_dimension(),
            'max_sequence_length': self.max_sequence_length,
            'cache_size': len(self._memory_cache)
        }

class TextPreprocessor:
    """Pré-processador de texto para embeddings"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Limpa e normaliza texto"""
        if not text:
            return ""
        
        # Remove caracteres especiais excessivos
        import re
        text = re.sub(r'\s+', ' ', text)  # Múltiplos espaços
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)  # Mantém pontuação básica
        
        # Normaliza
        text = text.strip().lower()
        
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = None) -> str:
        """Trunca texto para comprimento máximo"""
        max_length = max_length or 512
        
        if len(text) <= max_length:
            return text
        
        # Trunca em palavra completa
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Se tem espaço próximo ao final
            return truncated[:last_space]
        
        return truncated
    