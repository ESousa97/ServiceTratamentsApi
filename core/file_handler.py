import pandas as pd
import numpy as np
import chardet
import csv
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class FileMetadata:
    """Metadados do arquivo CSV"""
    filename: str
    file_size: int
    encoding: str
    delimiter: str
    rows_count: int
    columns_count: int
    column_names: List[str]
    column_types: Dict[str, str]
    has_header: bool
    estimated_memory_mb: float
    sample_data: List[Dict]

class CSVProcessor:
    """Processador inteligente de arquivos CSV"""
    
    def __init__(self):
        self.supported_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        self.common_delimiters = [',', ';', '\t', '|', ':']
        
    def detect_encoding(self, file_path: str, sample_size: int = 10000) -> str:
        """Detecta encoding do arquivo"""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(sample_size)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                
                # Verifica se o encoding detectado é suportado
                if encoding and encoding.lower() in [enc.lower() for enc in self.supported_encodings]:
                    return encoding
                
                # Fallback para encodings comuns
                for enc in self.supported_encodings:
                    try:
                        raw_data.decode(enc)
                        return enc
                    except:
                        continue
                        
                return 'utf-8'  # Default fallback
        except Exception as e:
            logger.warning(f"Erro na detecção de encoding: {e}")
            return 'utf-8'
    
    def detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detecta delimitador do CSV"""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                # Lê algumas linhas para análise
                sample_lines = [file.readline() for _ in range(min(10, sum(1 for _ in file)))]
                file.seek(0)
                sample_text = ''.join(sample_lines)
            
            # Usa o detector do Python
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample_text, delimiters=''.join(self.common_delimiters))
                return dialect.delimiter
            except:
                pass
            
            # Fallback: conta ocorrências de delimitadores
            delimiter_counts = {}
            for delimiter in self.common_delimiters:
                count = sample_text.count(delimiter)
                if count > 0:
                    delimiter_counts[delimiter] = count
            
            if delimiter_counts:
                return max(delimiter_counts, key=delimiter_counts.get)
            
            return ','  # Default
            
        except Exception as e:
            logger.warning(f"Erro na detecção de delimitador: {e}")
            return ','
    
    def detect_header(self, file_path: str, encoding: str, delimiter: str) -> bool:
        """Detecta se arquivo tem cabeçalho"""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                sample_lines = [file.readline().strip() for _ in range(min(3, sum(1 for _ in file)))]
            
            if len(sample_lines) < 2:
                return False
            
            first_row = sample_lines[0].split(delimiter)
            second_row = sample_lines[1].split(delimiter)
            
            # Verifica se primeira linha tem tipos diferentes da segunda
            first_numeric = sum(1 for cell in first_row if self._is_numeric(cell.strip()))
            second_numeric = sum(1 for cell in second_row if self._is_numeric(cell.strip()))
            
            # Se primeira linha tem menos números, provavelmente é header
            if first_numeric < second_numeric and first_numeric / len(first_row) < 0.5:
                return True
            
            # Verifica padrões de nomes de colunas
            header_patterns = ['id', 'name', 'data', 'valor', 'quantidade', 'date', 'time']
            header_like = sum(1 for cell in first_row 
                            if any(pattern in cell.lower() for pattern in header_patterns))
            
            return header_like > len(first_row) * 0.3
            
        except Exception as e:
            logger.warning(f"Erro na detecção de header: {e}")
            return True  # Assume que tem header por padrão
    
    def _is_numeric(self, value: str) -> bool:
        """Verifica se valor é numérico"""
        try:
            float(value.replace(',', '.'))
            return True
        except:
            return False
    
    def analyze_file(self, file_path: str) -> FileMetadata:
        """Analisa arquivo CSV e retorna metadados"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        file_size = file_path.stat().st_size
        encoding = self.detect_encoding(str(file_path))
        delimiter = self.detect_delimiter(str(file_path), encoding)
        has_header = self.detect_header(str(file_path), encoding, delimiter)
        
        # Análise inicial com pandas
        try:
            # Lê apenas uma amostra para análise rápida
            sample_df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                header=0 if has_header else None,
                nrows=100
            )
            
            rows_count = self._count_lines(str(file_path)) - (1 if has_header else 0)
            columns_count = len(sample_df.columns)
            
            # Gera nomes de colunas se necessário
            if has_header:
                column_names = sample_df.columns.tolist()
            else:
                column_names = [f"Column_{i+1}" for i in range(columns_count)]
                sample_df.columns = column_names
            
            # Detecta tipos de dados
            column_types = self._detect_column_types(sample_df)
            
            # Estima uso de memória
            estimated_memory_mb = self._estimate_memory_usage(
                rows_count, columns_count, column_types
            )
            
            # Dados de amostra
            sample_data = sample_df.head(5).to_dict('records')
            
            return FileMetadata(
                filename=file_path.name,
                file_size=file_size,
                encoding=encoding,
                delimiter=delimiter,
                rows_count=rows_count,
                columns_count=columns_count,
                column_names=column_names,
                column_types=column_types,
                has_header=has_header,
                estimated_memory_mb=estimated_memory_mb,
                sample_data=sample_data
            )
            
        except Exception as e:
            logger.error(f"Erro na análise do arquivo: {e}")
            raise
    
    def _count_lines(self, file_path: str) -> int:
        """Conta linhas do arquivo eficientemente"""
        try:
            with open(file_path, 'rb') as file:
                count = sum(1 for _ in file)
            return count
        except:
            return 0
    
    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detecta tipos de dados das colunas"""
        column_types = {}
        
        for column in df.columns:
            series = df[column].dropna()
            
            if len(series) == 0:
                column_types[column] = 'object'
                continue
            
            # Tenta converter para numérico
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isna().all():
                if (numeric_series % 1 == 0).all():
                    column_types[column] = 'integer'
                else:
                    column_types[column] = 'float'
                continue
            
            # Tenta converter para datetime
            try:
                pd.to_datetime(series, errors='raise')
                column_types[column] = 'datetime'
                continue
            except:
                pass
            
            # Verifica se é categórico
            unique_ratio = len(series.unique()) / len(series)
            if unique_ratio < 0.1 and len(series.unique()) < 50:
                column_types[column] = 'category'
            else:
                column_types[column] = 'text'
        
        return column_types
    
    def _estimate_memory_usage(self, rows: int, columns: int, 
                             column_types: Dict[str, str]) -> float:
        """Estima uso de memória em MB"""
        bytes_per_type = {
            'integer': 8,
            'float': 8,
            'datetime': 8,
            'category': 4,
            'text': 50,  # Estimativa média
            'object': 50
        }
        
        total_bytes = 0
        for col_type in column_types.values():
            bytes_per_cell = bytes_per_type.get(col_type, 50)
            total_bytes += rows * bytes_per_cell
        
        # Adiciona overhead do pandas
        total_bytes *= 1.5
        
        return total_bytes / (1024 * 1024)  # Converte para MB
    
    def load_data_chunked(self, file_path: str, metadata: FileMetadata, 
                         chunk_size: int = None) -> pd.DataFrame:
        """Carrega dados em chunks para economizar memória"""
        chunk_size = chunk_size or config.CHUNK_SIZE
        
        chunks = []
        for chunk in pd.read_csv(
            file_path,
            encoding=metadata.encoding,
            delimiter=metadata.delimiter,
            header=0 if metadata.has_header else None,
            chunksize=chunk_size,
            low_memory=False
        ):
            if not metadata.has_header:
                chunk.columns = metadata.column_names
            
            # Aplica tipos de dados otimizados
            chunk = self._optimize_dtypes(chunk, metadata.column_types)
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    def _optimize_dtypes(self, df: pd.DataFrame, 
                        column_types: Dict[str, str]) -> pd.DataFrame:
        """Otimiza tipos de dados para economizar memória"""
        for column, col_type in column_types.items():
            if column not in df.columns:
                continue
                
            try:
                if col_type == 'integer':
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                elif col_type == 'float':
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('float32')
                elif col_type == 'datetime':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif col_type == 'category':
                    df[column] = df[column].astype('category')
                # text e object mantêm como object (padrão)
            except Exception as e:
                logger.warning(f"Erro ao otimizar tipo da coluna {column}: {e}")
        
        return df

class FileValidator:
    """Validador de arquivos"""
    
    @staticmethod
    def validate_file(file_path: str, metadata: FileMetadata) -> List[str]:
        """Valida arquivo e retorna lista de problemas encontrados"""
        issues = []
        
        # Verifica tamanho do arquivo
        if metadata.file_size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            issues.append(f"Arquivo muito grande: {metadata.file_size / (1024*1024):.1f}MB")
        
        # Verifica uso estimado de memória
        if metadata.estimated_memory_mb > config.MAX_MEMORY_GB * 1024:
            issues.append(f"Uso estimado de memória muito alto: {metadata.estimated_memory_mb:.1f}MB")
        
        # Verifica se tem dados suficientes
        if metadata.rows_count < 2:
            issues.append("Arquivo deve ter pelo menos 2 linhas de dados")
        
        if metadata.columns_count < 1:
            issues.append("Arquivo deve ter pelo menos 1 coluna")
        
        # Verifica nomes de colunas duplicados
        if len(metadata.column_names) != len(set(metadata.column_names)):
            issues.append("Existem nomes de colunas duplicados")
        
        return issues
    