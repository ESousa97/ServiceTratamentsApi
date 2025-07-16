import psutil
import gc
import threading
import time
from typing import Callable, Any, List, Generator
from functools import wraps
import logging
from dataclasses import dataclass
from config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Estatísticas de memória"""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    process_memory_gb: float

class MemoryManager:
    """Gerenciador de memória para controlar uso de RAM"""
    
    def __init__(self, max_memory_gb: float = None):
        self.max_memory_gb = max_memory_gb or config.MAX_MEMORY_GB
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        self._lock = threading.Lock()
        
    def get_memory_stats(self) -> MemoryStats:
        """Retorna estatísticas atuais de memória"""
        # Memória do sistema
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_gb = memory.used / (1024**3)
        percent_used = memory.percent
        
        # Memória do processo atual
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024**3)
        
        return MemoryStats(
            total_gb=total_gb,
            available_gb=available_gb,
            used_gb=used_gb,
            percent_used=percent_used,
            process_memory_gb=process_memory_gb
        )
    
    def check_memory_limit(self) -> bool:
        """Verifica se está dentro do limite de memória"""
        stats = self.get_memory_stats()
        return stats.process_memory_gb <= self.max_memory_gb
    
    def get_available_memory_gb(self) -> float:
        """Retorna memória disponível para uso"""
        stats = self.get_memory_stats()
        max_usable = min(self.max_memory_gb, stats.available_gb * 0.8)
        current_used = stats.process_memory_gb
        return max(0, max_usable - current_used)
    
    def memory_limiter(self, func: Callable) -> Callable:
        """Decorator para limitar uso de memória de funções"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Verifica memória antes da execução
            if not self.check_memory_limit():
                self.force_garbage_collection()
                if not self.check_memory_limit():
                    raise MemoryError(f"Limite de memória excedido: {self.get_memory_stats().process_memory_gb:.2f}GB")
            
            try:
                result = func(*args, **kwargs)
                
                # Verifica memória após execução
                if not self.check_memory_limit():
                    logger.warning("Limite de memória excedido após execução da função")
                    self.force_garbage_collection()
                
                return result
            except MemoryError as e:
                logger.error(f"Erro de memória em {func.__name__}: {e}")
                self.force_garbage_collection()
                raise
                
        return wrapper
    
    def force_garbage_collection(self):
        """Força coleta de lixo agressiva"""
        logger.info("Executando coleta de lixo...")
        for i in range(3):
            gc.collect()
        
        stats = self.get_memory_stats()
        logger.info(f"Memória após GC: {stats.process_memory_gb:.2f}GB")
    
    def chunk_data(self, data: List[Any], max_chunk_size: int = None) -> Generator[List[Any], None, None]:
        """Divide dados em chunks baseado na memória disponível"""
        if max_chunk_size is None:
            available_gb = self.get_available_memory_gb()
            # Estima itens por GB (ajustar conforme necessário)
            estimated_items_per_gb = 10000
            max_chunk_size = max(100, int(available_gb * estimated_items_per_gb))
        
        for i in range(0, len(data), max_chunk_size):
            chunk = data[i:i + max_chunk_size]
            
            # Verifica memória antes de retornar chunk
            if not self.check_memory_limit():
                self.force_garbage_collection()
            
            yield chunk
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Inicia monitoramento contínuo de memória"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval_seconds,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Monitoramento de memória iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento de memória"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoramento de memória parado")
    
    def _monitor_loop(self, interval_seconds: int):
        """Loop de monitoramento de memória"""
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Log estatísticas
                logger.debug(f"Memória: {stats.process_memory_gb:.2f}GB / {self.max_memory_gb}GB")
                
                # Verifica limite
                if stats.process_memory_gb > self.max_memory_gb * 0.9:
                    logger.warning(f"Memória próxima do limite: {stats.process_memory_gb:.2f}GB")
                    self.force_garbage_collection()
                
                # Chama callbacks registrados
                for callback in self.callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"Erro em callback de monitoramento: {e}")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Erro no monitoramento de memória: {e}")
                time.sleep(interval_seconds)
    
    def add_memory_callback(self, callback: Callable[[MemoryStats], None]):
        """Adiciona callback para monitoramento de memória"""
        with self._lock:
            self.callbacks.append(callback)
    
    def remove_memory_callback(self, callback: Callable[[MemoryStats], None]):
        """Remove callback de monitoramento"""
        with self._lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

class ChunkProcessor:
    """Processador que trabalha com chunks de dados"""
    
    def __init__(self, memory_manager: MemoryManager = None):
        self.memory_manager = memory_manager or MemoryManager()
    
    def process_in_chunks(self, data: List[Any], 
                         processor_func: Callable[[List[Any]], Any],
                         chunk_size: int = None,
                         combine_func: Callable[[List[Any]], Any] = None) -> Any:
        """Processa dados em chunks"""
        results = []
        
        for chunk in self.memory_manager.chunk_data(data, chunk_size):
            try:
                # Aplica função de processamento no chunk
                chunk_result = processor_func(chunk)
                results.append(chunk_result)
                
                # Log progresso
                logger.debug(f"Processado chunk de {len(chunk)} itens")
                
            except Exception as e:
                logger.error(f"Erro no processamento de chunk: {e}")
                raise
        
        # Combina resultados se função fornecida
        if combine_func and results:
            return combine_func(results)
        
        return results

class MemoryOptimizer:
    """Otimizador de uso de memória"""
    
    @staticmethod
    def optimize_dataframe(df, categorical_threshold: float = 0.1):
        """Otimiza DataFrame para usar menos memória"""
        import pandas as pd
        
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Otimiza tipos numéricos
            if col_type in ['int64', 'int32']:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type in ['float64', 'float32']:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Converte para categórico se apropriado
            elif col_type == 'object':
                num_unique = df[col].nunique()
                num_total = len(df[col])
                
                if num_unique / num_total < categorical_threshold:
                    df[col] = df[col].astype('category')
        
        memory_after = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (memory_before - memory_after) / memory_before * 100
        
        logger.info(f"Otimização de memória: {memory_before:.1f}MB -> {memory_after:.1f}MB ({reduction:.1f}% redução)")
        
        return df
    
    @staticmethod
    def suggest_chunk_size(total_items: int, available_memory_gb: float, 
                          item_size_bytes: int = 1000) -> int:
        """Sugere tamanho de chunk baseado na memória disponível"""
        available_bytes = available_memory_gb * 1024**3 * 0.5  # Usa 50% da memória disponível
        suggested_size = int(available_bytes / item_size_bytes)
        
        # Limites mínimo e máximo
        min_size = 100
        max_size = min(50000, total_items)
        
        return max(min_size, min(suggested_size, max_size))

# Instância global
memory_manager = MemoryManager()
