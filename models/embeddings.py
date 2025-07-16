import numpy as np
import hashlib
import pickle
from typing import List, Dict, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import logging

# Importações locais - SEM instâncias globais aqui
from core.memory_manager import memory_manager
from config.settings import config
from config.database import db, cache

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Engine para geração e cache de embeddings"""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self._load_model()
        
    def _load_model(self):
        """Carrega modelo de embeddings"""
        try:
            logger.info(f"Carregando modelo: {self.model_name}")
            
            # Tenta carregar com sentence-transformers primeiro
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.model_type = 'sentence_transformer'
                logger.info("Modelo carregado com sentence-transformers")
            except Exception as e:
                logger.warning(f"Falha no sentence-transformers, tentando transformers puro: {e}")
                # Fallback para transformers
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model_type = 'transformer'
                logger.info("Modelo carregado com transformers")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            # Fallback para modelo simples
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Carrega modelo fallback simples"""
        try:
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=self.device)
            self.model_type = 'sentence_transformer'
            self.model_name = 'paraphrase-MiniLM-L6-v2'
            logger.warning("Usando modelo fallback: paraphrase-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo fallback: {e}")
            raise RuntimeError("Não foi possível carregar nenhum modelo de embeddings")
    
    def _generate_text_hash(self, text: str) -> str:
        """Gera hash único para texto"""
        text_normalized = text.strip().lower()
        hash_input = f"{text_normalized}_{self.model_name}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Recupera embedding do cache"""
        text_hash = self._generate_text_hash(text)
        
        # Tenta cache em memória primeiro
        cached = cache.get(f"embedding_{text_hash}")
        if cached is not None:
            return cached
        
        # Tenta banco de dados
        try:
            with db.get_cursor() as cursor:
                cursor.execute('''
                    SELECT embedding FROM embeddings_cache 
                    WHERE text_hash = %s AND model_name = %s
                ''', (text_hash, self.model_name))
                
                row = cursor.fetchone()
                if row:
                    embedding = pickle.loads(row[0])
                    # Adiciona ao cache em memória
                    cache.set(f"embedding_{text_hash}", embedding, expire_seconds=3600)
                    return embedding
        except Exception as e:
            logger.warning(f"Erro ao recuperar embedding do cache: {e}")
        
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Armazena embedding no cache"""
        text_hash = self._generate_text_hash(text)
        
        # Cache em memória
        cache.set(f"embedding_{text_hash}", embedding, expire_seconds=3600)
        
        # Cache no banco de dados
        try:
            with db.get_cursor() as cursor:
                cursor.execute('''
                    INSERT OR REPLACE INTO embeddings_cache 
                    (text_hash, embedding, model_name)
                    VALUES (%s, %s, %s)
                ''', (text_hash, pickle.dumps(embedding), self.model_name))
        except Exception as e:
            logger.warning(f"Erro ao armazenar embedding no cache: {e}")
    
    @memory_manager.memory_limiter
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
            else:
                embedding = self._generate_with_transformer(text)
            
            # Armazena no cache
            if use_cache:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            # Retorna embedding zero em caso de erro
            return np.zeros(self._get_embedding_dimension())
    
    def _generate_with_transformer(self, text: str) -> np.ndarray:
        """Gera embedding usando transformers direto"""
        try:
            # Tokeniza texto
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                max_length=config.MAX_SEQUENCE_LENGTH,
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
            raise
    
    def generate_embeddings_batch(self, texts: List[str], 
                                 batch_size: int = None, 
                                 use_cache: bool = True) -> List[np.ndarray]:
        """Gera embeddings para lista de textos"""
        batch_size = batch_size or config.BATCH_SIZE
        
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
                    if self.model_type == 'sentence_transformer':
                        new_embeddings = self.model.encode(
                            uncached_texts, 
                            convert_to_numpy=True,
                            batch_size=min(batch_size, len(uncached_texts))
                        )
                    else:
                        new_embeddings = [
                            self._generate_with_transformer(text) 
                            for text in uncached_texts
                        ]
                    
                    # Adiciona novos embeddings aos resultados
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        batch_embeddings.append((idx, embedding))
                        
                        # Cache individual
                        if use_cache:
                            self._cache_embedding(uncached_texts[uncached_indices.index(idx)], embedding)
                    
                except Exception as e:
                    logger.error(f"Erro no batch {i//batch_size + 1}: {e}")
                    # Adiciona embeddings zero para textos com erro
                    zero_embedding = np.zeros(self._get_embedding_dimension())
                    for idx in uncached_indices:
                        batch_embeddings.append((idx, zero_embedding))
            
            # Ordena embeddings pela posição original
            batch_embeddings.sort(key=lambda x: x[0])
            all_embeddings.extend([emb for _, emb in batch_embeddings])
            
            logger.debug(f"Processado batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
        
        return all_embeddings
    
    def _get_embedding_dimension(self) -> int:
        """Retorna dimensão dos embeddings"""
        if self.model_type == 'sentence_transformer':
            return self.model.get_sentence_embedding_dimension()
        else:
            # Para transformers, faz uma inferência teste
            try:
                test_embedding = self._generate_with_transformer("test")
                return len(test_embedding)
            except:
                return 768  # Dimensão padrão para BERT-like models
    
    def get_model_info(self) -> Dict[str, any]:
        """Retorna informações do modelo"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'device': self.device,
            'embedding_dimension': self._get_embedding_dimension(),
            'max_sequence_length': config.MAX_SEQUENCE_LENGTH
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
        max_length = max_length or config.MAX_SEQUENCE_LENGTH
        
        if len(text) <= max_length:
            return text
        
        # Trunca em palavra completa
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Se tem espaço próximo ao final
            return truncated[:last_space]
        
        return truncated

# REMOVIDO: Não instancie nada aqui para evitar imports circulares!
# A instância será criada onde for necessária
