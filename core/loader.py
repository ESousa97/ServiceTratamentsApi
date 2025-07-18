import pandas as pd
import chardet
import os

def load_and_clean_excel(file_path):
    """
    Carrega arquivo Excel e realiza limpeza básica:
    - Remove linhas e colunas vazias
    - Limpa espaços e caracteres invisíveis em strings
    - Tenta converter colunas de datas
    """
    # Carrega com openpyxl (pode gerar warnings ignorados)
    df = pd.read_excel(file_path, engine='openpyxl')

    # Remove linhas e colunas totalmente vazias
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # Limpa espaços dos nomes das colunas
    df.columns = df.columns.str.strip()

    # Limpa strings nas colunas do tipo object
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r'[\x00-\x1F]+', '', regex=True)

    # Tenta converter colunas que parecem datas
    for col in df.columns:
        if 'data' in col.lower() or 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


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
            for chunk in pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize, low_memory=False):
                chunks.append(chunk)
                total_rows += len(chunk)
                if progress_callback:
                    progress_callback(total_rows)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, low_memory=False)
    elif ext in ['.xlsx', '.xls']:
        df = load_and_clean_excel(file_path)  # usa a função de limpeza aqui
        if progress_callback:
            progress_callback(len(df))
    else:
        raise ValueError('Formato de arquivo não suportado!')

    return df
