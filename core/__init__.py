"""
Módulo core do Intelligent CSV Processor
Contém as funcionalidades principais de processamento de arquivos e gerenciamento
"""

from .file_handler import CSVProcessor, FileValidator, FileMetadata
from .memory_manager import MemoryManager, memory_manager
from .semantic_engine import SemanticEngine, semantic_processor
from .neural_processor import NeuralProcessor

__all__ = [
    'CSVProcessor', 'FileValidator', 'FileMetadata',
    'MemoryManager', 'memory_manager',
    'SemanticEngine', 'semantic_processor',
    'NeuralProcessor'
]
