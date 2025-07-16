"""
Módulo de modelos de Machine Learning do Intelligent CSV Processor
Contém redes neurais, embeddings e engines de similaridade
"""

# REMOVIDO: Não importa instâncias para evitar imports circulares
from .embeddings import EmbeddingEngine
from .neural_network import TextClassifier, AnomalyDetector
from .similarity_engine import SimilarityEngine

__all__ = [
    'EmbeddingEngine',  # REMOVIDO: 'embedding_engine' 
    'TextClassifier', 'AnomalyDetector',
    'SimilarityEngine'
]
