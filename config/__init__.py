"""
Módulo de configuração do Intelligent CSV Processor
Contém configurações globais, gerenciamento de banco de dados e cache
"""

from .settings import config, ConfigManager
from .database import db, cache

__all__ = ['config', 'ConfigManager', 'db', 'cache']