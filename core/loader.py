# core/loader.py
import pandas as pd
import chardet
import os

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(2048))
    return result['encoding'] or 'utf-8'

def detect_delimiter(file_path, encoding):
    with open(file_path, 'r', encoding=encoding) as f:
        sample = f.read(4096)
        delimiters = [',', ';', '\t', '|', ':']
        delimiter_scores = {d: sample.count(d) for d in delimiters}
        return max(delimiter_scores, key=delimiter_scores.get)

def load_spreadsheet(file_path, chunksize=None, progress_callback=None):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        encoding = detect_encoding(file_path)
        delimiter = detect_delimiter(file_path, encoding)
        if chunksize:
            chunks = []
            total_rows = 0
            for chunk in pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
                chunks.append(chunk)
                total_rows += len(chunk)
                if progress_callback:
                    progress_callback(total_rows)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
    elif ext in ['.xlsx', '.xls']:
        # Excel não suporta leitura em chunks no pandas
        df = pd.read_excel(file_path)
        if progress_callback:
            progress_callback(len(df))
    else:
        raise ValueError('Formato de arquivo não suportado!')
    return df
