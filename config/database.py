# config/database.py

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
import logging

logger = logging.getLogger(__name__)

# FUNÇÃO PARA LIMPEZA JSON - VERSÃO MELHORADA
def clean_for_json_improved(obj):
    """Versão melhorada da limpeza JSON - especialmente para dashboards"""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            try:
                cleaned[str(k)] = clean_for_json_improved(v)
            except Exception as e:
                logger.warning(f"Erro ao limpar chave {k}: {e}")
                cleaned[str(k)] = str(v)
        return cleaned
    elif isinstance(obj, (list, tuple)):
        cleaned = []
        for item in obj:
            try:
                cleaned.append(clean_for_json_improved(item))
            except Exception as e:
                logger.warning(f"Erro ao limpar item da lista: {e}")
                cleaned.append(str(item))
        return cleaned
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        val = float(obj)
        # Trata valores especiais
        if np.isnan(val):
            return None
        elif np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except:
            return str(obj)
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        try:
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        except:
            return str(obj)
    elif isinstance(obj, datetime):
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
    elif isinstance(obj, str):
        # Limita strings muito longas
        if len(obj) > 10000:
            return obj[:10000] + "... [truncated]"
        return obj
    elif isinstance(obj, (int, float, bool)):
        return obj
    else:
        # Para qualquer outro tipo, converte para string
        try:
            return str(obj)
        except:
            return "<<unserializable>>"

# FUNÇÃO ORIGINAL PARA BACKUP / OUTRAS SITUAÇÕES
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

    # SUBSTITUIR PELO MÉTODO MELHORADO save_analysis_improved
    def save_analysis(self, job_id: int, analysis_type: str, results: Dict):
        """Versão melhorada para salvar análises"""
        with self.get_cursor() as cursor:
            try:
                # Limpeza especial para dashboards
                if analysis_type == 'dashboard':
                    cleaned_results = self._clean_dashboard_data(results)
                else:
                    cleaned_results = clean_for_json_improved(results)
                
                # Tenta serializar
                json_results = json.dumps(cleaned_results, ensure_ascii=False, separators=(',', ':'))
                
                # Verifica tamanho (limite de 16MB para JSON)
                if len(json_results.encode('utf-8')) > 16 * 1024 * 1024:
                    # Se muito grande, cria versão resumida
                    cleaned_results = self._create_summary_version(analysis_type, results)
                    json_results = json.dumps(cleaned_results, ensure_ascii=False, separators=(',', ':'))
                
                cursor.execute('''
                    INSERT INTO analyses (job_id, analysis_type, results)
                    VALUES (?, ?, ?)
                ''', (job_id, analysis_type, json_results))
                
                logger.info(f"Análise {analysis_type} salva com sucesso para job {job_id}")
                
            except Exception as e:
                # Fallback: salva apenas informações essenciais
                fallback_results = {
                    'error': 'Serialization failed',
                    'original_error': str(e),
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat(),
                    'summary': self._extract_essential_info(results)
                }
                
                cursor.execute('''
                    INSERT INTO analyses (job_id, analysis_type, results)
                    VALUES (?, ?, ?)
                ''', (job_id, analysis_type, json.dumps(fallback_results)))
                
                logger.error(f"Erro na serialização de {analysis_type}, usando fallback: {e}")

    def _clean_dashboard_data(self, dashboard_data: Dict) -> Dict:
        """Limpeza especial para dados de dashboard"""
        try:
            cleaned = {}
            
            # Metadata
            if 'metadata' in dashboard_data:
                cleaned['metadata'] = clean_for_json_improved(dashboard_data['metadata'])
            
            # Summary cards
            if 'summary' in dashboard_data:
                cleaned['summary'] = clean_for_json_improved(dashboard_data['summary'])
            
            # Charts - requer cuidado especial
            if 'charts' in dashboard_data:
                cleaned_charts = []
                for chart in dashboard_data['charts']:
                    cleaned_chart = {
                        'title': chart.get('title', 'Sem título'),
                        'type': chart.get('type', 'unknown'),
                        'description': chart.get('description', ''),
                    }
                    
                    # Para chart_json, verifica se já é string ou precisa converter
                    if 'chart_json' in chart:
                        chart_json = chart['chart_json']
                        if isinstance(chart_json, str):
                            # Já é string JSON, mantém
                            cleaned_chart['chart_json'] = chart_json
                        else:
                            # Precisa converter para string
                            try:
                                cleaned_chart['chart_json'] = json.dumps(chart_json)
                            except:
                                cleaned_chart['chart_json'] = None
                                cleaned_chart['error'] = 'Failed to serialize chart'
                    
                    # Adiciona estatísticas se existirem
                    if 'stats' in chart:
                        cleaned_chart['stats'] = clean_for_json_improved(chart['stats'])
                    
                    cleaned_charts.append(cleaned_chart)
                
                cleaned['charts'] = cleaned_charts
            
            # Data preview
            if 'data_preview' in dashboard_data:
                preview = dashboard_data['data_preview']
                cleaned_preview = {
                    'columns': preview.get('columns', [])[:50],  # Limita colunas
                    'total_rows': preview.get('total_rows', 0),
                    'preview_rows': preview.get('preview_rows', 0)
                }
                
                # Limita dados do preview
                if 'data' in preview and isinstance(preview['data'], list):
                    cleaned_preview['data'] = preview['data'][:10]  # Máximo 10 linhas
                
                cleaned['data_preview'] = cleaned_preview
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Erro na limpeza de dashboard: {e}")
            return {
                'error': 'Dashboard cleaning failed',
                'original_error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _create_summary_version(self, analysis_type: str, results: Dict) -> Dict:
        """Cria versão resumida quando dados são muito grandes"""
        summary = {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'summary': 'Data too large - summary version created'
        }
        
        if analysis_type == 'dashboard':
            summary.update({
                'metadata': results.get('metadata', {}),
                'charts_count': len(results.get('charts', [])),
                'summary_cards_count': len(results.get('summary', [])),
                'has_data_preview': 'data_preview' in results
            })
        elif analysis_type == 'statistical':
            summary.update({
                'basic_stats': results.get('basic_stats', {}),
                'data_types_count': len(results.get('data_types', {})),
                'quality_score': results.get('data_quality', {}).get('overall_score', 0)
            })
        
        return summary

    def _extract_essential_info(self, results: Dict) -> Dict:
        """Extrai informações essenciais de qualquer análise"""
        essential = {
            'type': 'summary',
            'timestamp': datetime.now().isoformat()
        }
        
        # Tenta extrair informações básicas
        try:
            if isinstance(results, dict):
                # Conta elementos principais
                essential['keys_count'] = len(results.keys())
                essential['main_keys'] = list(results.keys())[:10]
                
                # Extrai valores simples
                for key, value in results.items():
                    if isinstance(value, (int, float, str, bool)) and len(str(value)) < 100:
                        essential[f'simple_{key}'] = value
        except:
            pass
        
        return essential

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
