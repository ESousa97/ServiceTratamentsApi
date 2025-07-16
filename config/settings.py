import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BaseConfig:
    """Configuração base da aplicação"""
    # Configurações de memória
    MAX_MEMORY_GB: float = 8.0
    CHUNK_SIZE: int = 10000
    BATCH_SIZE: int = 512
    
    # Configurações de arquivos
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: set = None
    UPLOAD_FOLDER: str = "uploads"
    TEMP_FOLDER: str = "temp"
    OUTPUT_FOLDER: str = "outputs"
    
    # Configurações da rede neural
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MAX_SEQUENCE_LENGTH: int = 512
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Configurações de processamento
    N_WORKERS: int = 4
    TIMEOUT_SECONDS: int = 3600
    PROGRESS_UPDATE_INTERVAL: int = 5
    
    # Configurações do banco de dados
    DATABASE_URL: str = "sqlite:///data.db"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Configurações da API
    SECRET_KEY: str = "dev-secret-key"
    DEBUG: bool = True
    CORS_ORIGINS: list = None
    
    def __post_init__(self):
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.tsv'}
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ['*']

@dataclass 
class DevelopmentConfig(BaseConfig):
    """Configuração para desenvolvimento"""
    DEBUG: bool = True
    MAX_MEMORY_GB: float = 4.0
    N_WORKERS: int = 2
    SECRET_KEY: str = "dev-secret-key-change-in-production"

@dataclass
class ProductionConfig(BaseConfig):
    """Configuração para produção"""
    DEBUG: bool = False
    MAX_MEMORY_GB: float = 8.0
    N_WORKERS: int = 8
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'production-secret-key')
    DATABASE_URL: str = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost/db')
    REDIS_URL: str = os.environ.get('REDIS_URL', 'redis://localhost:6379')

@dataclass
class TestConfig(BaseConfig):
    """Configuração para testes"""
    DEBUG: bool = True
    MAX_MEMORY_GB: float = 1.0
    CHUNK_SIZE: int = 100
    BATCH_SIZE: int = 32
    DATABASE_URL: str = "sqlite:///:memory:"

class ConfigManager:
    """Gerenciador de configurações"""
    
    _configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestConfig
    }
    
    @classmethod
    def get_config(cls, env: str = None) -> BaseConfig:
        """Retorna a configuração baseada no ambiente"""
        if env is None:
            env = os.environ.get('FLASK_ENV', 'development')
        
        config_class = cls._configs.get(env, DevelopmentConfig)
        return config_class()
    
    @classmethod
    def create_directories(cls, config: BaseConfig):
        """Cria diretórios necessários"""
        directories = [
            config.UPLOAD_FOLDER,
            config.TEMP_FOLDER,
            config.OUTPUT_FOLDER,
            'logs',
            'models/cache'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Configuração global
config = ConfigManager.get_config()
