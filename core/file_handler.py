import pandas as pd
import numpy as np
import chardet
import csv
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import tempfile
import shutil

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

    def _clean_file_for_processing(self, file_path: str, encoding: str) -> str:
        """Remove caracteres problemáticos do arquivo e retorna caminho do arquivo limpo"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
            with open(file_path, 'r', encoding=encoding, errors='replace') as input_file:
                with open(temp_fd, 'w', encoding='utf-8', newline='') as output_file:
                    for line_num, line in enumerate(input_file, 1):
                        cleaned_line = line.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
                        cleaned_line = ''.join(char for char in cleaned_line
                                               if ord(char) >= 32 or char in ['\n', '\t'])
                        if cleaned_line.strip():
                            output_file.write(cleaned_line)
                        elif line_num <= 5:
                            output_file.write('\n')
            logger.info(f"Arquivo limpo criado: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Erro na limpeza do arquivo: {e}")
            return file_path

    def analyze_file(self, file_path: str) -> FileMetadata:
        """Analisa arquivo CSV e retorna metadados - VERSÃO COM LIMPEZA"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        file_size = file_path.stat().st_size
        encoding = self.detect_encoding(str(file_path))

        cleaned_file_path = self._clean_file_for_processing(str(file_path), encoding)

        try:
            delimiter = self.detect_delimiter(cleaned_file_path, 'utf-8')
            has_header = self.detect_header(cleaned_file_path, 'utf-8', delimiter)

            sample_df = None
            final_delimiter = delimiter
            final_has_header = has_header

            try:
                sample_df = pd.read_csv(
                    cleaned_file_path,
                    encoding='utf-8',
                    delimiter=delimiter,
                    header=0 if has_header else None,
                    nrows=100,
                    on_bad_lines='skip',
                    engine='python'
                )
                logger.info("Arquivo lido com configurações detectadas")
            except Exception as e1:
                logger.warning(f"Primeira tentativa falhou: {e1}")
                try:
                    sample_df = pd.read_csv(
                        cleaned_file_path,
                        encoding='utf-8',
                        nrows=100,
                        on_bad_lines='skip',
                        sep=None,
                        engine='c',
                        low_memory=False
                    )
                    logger.info("Arquivo lido com detecção automática do pandas")
                    final_delimiter = ','
                except Exception as e2:
                    logger.warning(f"Segunda tentativa falhou: {e2}")
                    try:
                        sample_df = pd.read_csv(
                            cleaned_file_path,
                            encoding='utf-8',
                            delimiter=',',
                            header=None,
                            nrows=100,
                            on_bad_lines='skip',
                            engine='c'
                        )
                        final_delimiter = ','
                        final_has_header = False
                        logger.info("Arquivo lido com configuração forçada (vírgula, sem header)")
                    except Exception as e3:
                        logger.error(f"Todas as tentativas falharam: {e3}")
                        raise ValueError(f"Não foi possível ler o arquivo CSV: {e3}")

            if sample_df is None or len(sample_df) == 0:
                raise ValueError("Arquivo CSV vazio ou não pôde ser processado")

            rows_count = self._count_lines(str(file_path)) - (1 if final_has_header else 0)
            columns_count = len(sample_df.columns)

            if final_has_header and not sample_df.columns.empty:
                column_names = sample_df.columns.tolist()
            else:
                column_names = [f"Column_{i+1}" for i in range(columns_count)]
                sample_df.columns = column_names

            column_types = self._detect_column_types(sample_df)
            estimated_memory_mb = self._estimate_memory_usage(
                rows_count, columns_count, column_types
            )
            try:
                sample_data = sample_df.head(5).to_dict('records')
            except Exception as e:
                logger.warning(f"Erro ao gerar dados de amostra: {e}")
                sample_data = []

            metadata = FileMetadata(
                filename=file_path.name,
                file_size=file_size,
                encoding='utf-8',
                delimiter=final_delimiter,
                rows_count=rows_count,
                columns_count=columns_count,
                column_names=column_names,
                column_types=column_types,
                has_header=final_has_header,
                estimated_memory_mb=estimated_memory_mb,
                sample_data=sample_data
            )

            metadata._cleaned_file_path = cleaned_file_path
            return metadata

        finally:
            if cleaned_file_path != str(file_path) and Path(cleaned_file_path).exists():
                try:
                    # Não remove ainda - será usado no load_data_chunked
                    pass
                except:
                    pass

    def load_data_chunked(self, file_path: str, metadata: FileMetadata,
                         chunk_size: int = None) -> pd.DataFrame:
        """Carrega dados em chunks para economizar memória - VERSÃO CORRIGIDA"""
        chunk_size = chunk_size or config.CHUNK_SIZE
        actual_file_path = getattr(metadata, '_cleaned_file_path', file_path)
        actual_encoding = 'utf-8' if hasattr(metadata, '_cleaned_file_path') else metadata.encoding
        chunks = []

        try:
            read_params = {
                'encoding': actual_encoding,
                'delimiter': metadata.delimiter,
                'header': 0 if metadata.has_header else None,
                'chunksize': chunk_size,
                'on_bad_lines': 'skip'
            }
            if metadata.file_size > 50 * 1024 * 1024:
                read_params['engine'] = 'c'
                read_params['low_memory'] = False
            else:
                read_params['engine'] = 'python'
            for chunk in pd.read_csv(actual_file_path, **read_params):
                if not metadata.has_header:
                    chunk.columns = metadata.column_names
                chunk = self._clean_numeric_data(chunk)
                chunk = self._optimize_dtypes(chunk, metadata.column_types)
                chunks.append(chunk)
        except Exception as e:
            logger.error(f"Erro no carregamento chunked: {e}")
            try:
                read_params = {
                    'encoding': actual_encoding,
                    'delimiter': metadata.delimiter,
                    'header': 0 if metadata.has_header else None,
                    'on_bad_lines': 'skip',
                    'engine': 'c' if metadata.file_size > 50 * 1024 * 1024 else 'python'
                }
                if read_params['engine'] == 'c':
                    read_params['low_memory'] = False
                full_df = pd.read_csv(actual_file_path, **read_params)
                if not metadata.has_header:
                    full_df.columns = metadata.column_names
                full_df = self._clean_numeric_data(full_df)
                return self._optimize_dtypes(full_df, metadata.column_types)
            except Exception as e2:
                logger.error(f"Erro no fallback de carregamento: {e2}")
                raise

        if not chunks:
            raise ValueError("Nenhum chunk foi carregado com sucesso")
        final_df = pd.concat(chunks, ignore_index=True)
        if hasattr(metadata, '_cleaned_file_path') and Path(metadata._cleaned_file_path).exists():
            try:
                Path(metadata._cleaned_file_path).unlink()
                logger.info(f"Arquivo temporário limpo: {metadata._cleaned_file_path}")
            except:
                pass
        return final_df

    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dados numéricos problemáticos"""
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column] = df[column].replace([np.inf, -np.inf], np.nan)
            max_float32 = np.finfo(np.float32).max
            min_float32 = np.finfo(np.float32).min
            mask_too_large = df[column] > max_float32
            mask_too_small = df[column] < min_float32
            if mask_too_large.any() or mask_too_small.any():
                logger.warning(f"Coluna {column}: {mask_too_large.sum() + mask_too_small.sum()} valores extremos convertidos para NaN")
                df.loc[mask_too_large | mask_too_small, column] = np.nan
        return df

    def _optimize_dtypes(self, df: pd.DataFrame,
                        column_types: Dict[str, str]) -> pd.DataFrame:
        """Otimiza tipos de dados para economizar memória - VERSÃO SEGURA"""
        for column, col_type in column_types.items():
            if column not in df.columns:
                continue
            try:
                if col_type == 'integer':
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                elif col_type == 'float':
                    numeric_col = pd.to_numeric(df[column], errors='coerce')
                    max_val = numeric_col.max()
                    min_val = numeric_col.min()
                    if (pd.isna(max_val) or abs(max_val) < np.finfo(np.float32).max) and \
                       (pd.isna(min_val) or abs(min_val) < np.finfo(np.float32).max):
                        df[column] = numeric_col.astype('float32')
                    else:
                        df[column] = numeric_col.astype('float64')
                        logger.info(f"Coluna {column} mantida como float64 devido a valores grandes")
                elif col_type == 'datetime':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif col_type == 'category':
                    df[column] = df[column].astype('category')
                # text e object mantêm como object (padrão)
            except Exception as e:
                logger.warning(f"Erro ao otimizar tipo da coluna {column}: {e}")
        return df

    def detect_encoding(self, file_path: str, sample_size: int = 10000) -> str:
        """Detecta encoding do arquivo"""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(sample_size)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                if encoding and encoding.lower() in [enc.lower() for enc in self.supported_encodings]:
                    return encoding
                for enc in self.supported_encodings:
                    try:
                        raw_data.decode(enc)
                        return enc
                    except:
                        continue
                return 'utf-8'
        except Exception as e:
            logger.warning(f"Erro na detecção de encoding: {e}")
            return 'utf-8'

    def detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detecta delimitador do CSV"""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                sample_lines = []
                for i, line in enumerate(file):
                    if i >= 20:
                        break
                    sample_lines.append(line)
                if not sample_lines:
                    return ','
                sample_text = ''.join(sample_lines)
            sniffer = csv.Sniffer()
            try:
                delimiters = ',;\t|:~'
                dialect = sniffer.sniff(sample_text, delimiters=delimiters)
                detected_delimiter = dialect.delimiter
                if detected_delimiter in delimiters:
                    return detected_delimiter
            except Exception as e:
                logger.warning(f"Sniffer falhou: {e}")
            delimiter_candidates = [',', ';', '\t', '|', ':', '~']
            delimiter_scores = {}
            for delimiter in delimiter_candidates:
                score = 0
                consistent_count = None
                for line in sample_lines[:10]:
                    count = line.count(delimiter)
                    if count > 0:
                        if consistent_count is None:
                            consistent_count = count
                            score += 10
                        elif consistent_count == count:
                            score += 5
                        else:
                            score -= 2
                if score > 0:
                    delimiter_scores[delimiter] = score
            if delimiter_scores:
                best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
                logger.info(f"Delimiter detectado: '{best_delimiter}' (score: {delimiter_scores[best_delimiter]})")
                return best_delimiter
            return ','
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
            first_numeric = sum(1 for cell in first_row if self._is_numeric(cell.strip()))
            second_numeric = sum(1 for cell in second_row if self._is_numeric(cell.strip()))
            if first_numeric < second_numeric and first_numeric / len(first_row) < 0.5:
                return True
            header_patterns = ['id', 'name', 'data', 'valor', 'quantidade', 'date', 'time']
            header_like = sum(1 for cell in first_row
                              if any(pattern in cell.lower() for pattern in header_patterns))
            return header_like > len(first_row) * 0.3
        except Exception as e:
            logger.warning(f"Erro na detecção de header: {e}")
            return True

    def _is_numeric(self, value: str) -> bool:
        """Verifica se valor é numérico"""
        try:
            float(value.replace(',', '.'))
            return True
        except:
            return False

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
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isna().all():
                if (numeric_series % 1 == 0).all():
                    column_types[column] = 'integer'
                else:
                    column_types[column] = 'float'
                continue
            try:
                pd.to_datetime(series, errors='raise')
                column_types[column] = 'datetime'
                continue
            except:
                pass
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
            'text': 50,
            'object': 50
        }
        total_bytes = 0
        for col_type in column_types.values():
            bytes_per_cell = bytes_per_type.get(col_type, 50)
            total_bytes += rows * bytes_per_cell
        total_bytes *= 1.5
        return total_bytes / (1024 * 1024)

class FileValidator:
    """Validador de arquivos"""

    @staticmethod
    def validate_file(file_path: str, metadata: FileMetadata) -> List[str]:
        """Valida arquivo e retorna lista de problemas encontrados"""
        issues = []
        if metadata.file_size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            issues.append(f"Arquivo muito grande: {metadata.file_size / (1024*1024):.1f}MB")
        if metadata.estimated_memory_mb > config.MAX_MEMORY_GB * 1024:
            issues.append(f"Uso estimado de memória muito alto: {metadata.estimated_memory_mb:.1f}MB")
        if metadata.rows_count < 2:
            issues.append("Arquivo deve ter pelo menos 2 linhas de dados")
        if metadata.columns_count < 1:
            issues.append("Arquivo deve ter pelo menos 1 coluna")
        if len(metadata.column_names) != len(set(metadata.column_names)):
            issues.append("Existem nomes de colunas duplicados")
        return issues
