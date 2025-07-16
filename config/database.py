# Adicione essas linhas no INÍCIO do arquivo config/database.py

import sqlite3
import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import threading
import redis
import numpy as np
import pandas as pd
from config.settings import config

# FUNÇÃO PARA LIMPEZA JSON - ADICIONE ESTA FUNÇÃO
def clean_for_json(obj):
    """Limpa objeto recursivamente para serialização JSON"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(clean_for_json(list(obj)))
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, (datetime)):
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # numpy scalars
        try:
            return obj.item()
        except:
            return str(obj)
    elif hasattr(obj, 'tolist'):  # arrays
        try:
            return obj.tolist()
        except:
            return str(obj)
    elif pd.isna(obj):  # pandas NaN values
        return None
    else:
        return obj

class DatabaseManager:
    """Gerenciador de banco de dados"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or config.DATABASE_URL
        self._local = threading.local()
        self.init_database()
    
    @property
    def connection(self):
        """Retorna conexão thread-local"""
        if not hasattr(self._local, 'connection'):
            if self.db_url.startswith('sqlite'):
                db_path = self.db_url.replace('sqlite:///', '')
                self._local.connection = sqlite3.connect(db_path, check_same_thread=False)
                self._local.connection.row_factory = sqlite3.Row
            else:
                # Para PostgreSQL ou outros bancos
                raise NotImplementedError("Apenas SQLite suportado nesta versão")
        return self._local.connection
    
    def init_database(self):
        """Inicializa as tabelas do banco"""
        with self.get_cursor() as cursor:
            # Tabela de jobs de processamento
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    progress REAL DEFAULT 0.0,
                    estimated_duration INTEGER,
                    actual_duration INTEGER,
                    file_size INTEGER,
                    rows_count INTEGER,
                    columns_count INTEGER,
                    error_message TEXT,
                    results_path TEXT
                )
            ''')
            
            # Tabela de análises geradas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    analysis_type TEXT NOT NULL,
                    results TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs (id)
                )
            ''')
            
            # Tabela de cache de embeddings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Índices para performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_created ON processing_jobs(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings_cache(text_hash)')
    
    @contextmanager
    def get_cursor(self):
        """Context manager para operações de banco"""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e
        finally:
            cursor.close()
    
    def create_job(self, filename: str, file_size: int, 
                   rows_count: int = None, columns_count: int = None) -> int:
        """Cria um novo job de processamento"""
        with self.get_cursor() as cursor:
            cursor.execute('''
                INSERT INTO processing_jobs 
                (filename, file_size, rows_count, columns_count)
                VALUES (?, ?, ?, ?)
            ''', (filename, file_size, rows_count, columns_count))
            return cursor.lastrowid
    
    def update_job_status(self, job_id: int, status: str, 
                         progress: float = None, error_message: str = None):
        """Atualiza status do job"""
        with self.get_cursor() as cursor:
            updates = ['status = ?']
            params = [status]
            
            if progress is not None:
                updates.append('progress = ?')
                params.append(progress)
            
            if error_message is not None:
                updates.append('error_message = ?')
                params.append(error_message)
            
            if status == 'processing':
                updates.append('started_at = CURRENT_TIMESTAMP')
            elif status in ['completed', 'failed']:
                updates.append('completed_at = CURRENT_TIMESTAMP')
            
            params.append(job_id)
            
            cursor.execute(f'''
                UPDATE processing_jobs 
                SET {', '.join(updates)}
                WHERE id = ?
            ''', params)
    
    def get_job(self, job_id: int) -> Optional[Dict]:
        """Retorna informações do job"""
        with self.get_cursor() as cursor:
            cursor.execute('SELECT * FROM processing_jobs WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def save_analysis(self, job_id: int, analysis_type: str, results: Dict):
        """Salva resultado de análise - VERSÃO CORRIGIDA"""
        with self.get_cursor() as cursor:
            # ✅ CORREÇÃO: Limpa dados antes de serializar
            try:
                cleaned_results = clean_for_json(results)
                json_results = json.dumps(cleaned_results, ensure_ascii=False)
                
                cursor.execute('''
                    INSERT INTO analyses (job_id, analysis_type, results)
                    VALUES (?, ?, ?)
                ''', (job_id, analysis_type, json_results))
                
            except Exception as e:
                # Se ainda falhar, salva uma versão simplificada
                simplified_results = {
                    'error': 'Serialization failed',
                    'original_error': str(e),
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat()
                }
                cursor.execute('''
                    INSERT INTO analyses (job_id, analysis_type, results)
                    VALUES (?, ?, ?)
                ''', (job_id, analysis_type, json.dumps(simplified_results)))
                
                # Log do erro para debug
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Erro na serialização de análise {analysis_type}: {e}")
    
    def get_analyses(self, job_id: int) -> List[Dict]:
        """Retorna análises de um job"""
        with self.get_cursor() as cursor:
            cursor.execute('''
                SELECT analysis_type, results, created_at 
                FROM analyses WHERE job_id = ?
            ''', (job_id,))
            
            analyses = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                # Tenta fazer parse do JSON
                try:
                    row_dict['results'] = json.loads(row_dict['results'])
                except:
                    # Se falhar, mantém como string
                    pass
                analyses.append(row_dict)
            
            return analyses

class CacheManager:
    """Gerenciador de cache usando Redis"""
    
    def __init__(self, redis_url: str = None):
        try:
            self.redis_client = redis.from_url(redis_url or config.REDIS_URL)
            self.redis_client.ping()
            self.enabled = True
        except:
            self.enabled = False
            self.memory_cache = {}
    
    def get(self, key: str) -> Any:
        """Recupera valor do cache"""
        if self.enabled:
            try:
                data = self.redis_client.get(key)
                return pickle.loads(data) if data else None
            except:
                return None
        else:
            return self.memory_cache.get(key)
    
    def set(self, key: str, value: Any, expire_seconds: int = 3600):
        """Armazena valor no cache"""
        if self.enabled:
            try:
                self.redis_client.setex(key, expire_seconds, pickle.dumps(value))
            except:
                pass
        else:
            self.memory_cache[key] = value
    
    def delete(self, key: str):
        """Remove valor do cache"""
        if self.enabled:
            try:
                self.redis_client.delete(key)
            except:
                pass
        else:
            self.memory_cache.pop(key, None)
    
    def clear_pattern(self, pattern: str):
        """Remove chaves que correspondem ao padrão"""
        if self.enabled:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except:
                pass
        else:
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache[key]

# Instâncias globais
db = DatabaseManager()
cache = CacheManager()