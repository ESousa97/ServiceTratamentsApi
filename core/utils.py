# core/utils.py
import os
import pandas as pd
import re

def validate_file(file_path, max_rows, max_size_mb):
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    if size_mb > max_size_mb:
        raise ValueError(f"Arquivo excede o tamanho máximo permitido de {max_size_mb} MB.")
    return True

def normalize_cep_column(df, cep_cols=None):
    """
    Detecta e normaliza colunas de CEP no DataFrame.
    Converte valores para strings, remove '.0', mantém só números e preenche zeros à esquerda.
    """
    if cep_cols is None:
        cep_cols = []
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(100)
            cnt = sum(1 for val in sample if re.fullmatch(r'\d{7,8}(\.0)?', val))
            if cnt >= len(sample)*0.8:
                cep_cols.append(col)
    for col in cep_cols:
        def format_cep(val):
            if pd.isna(val):
                return val
            s = str(val)
            s = s.replace('.0','').replace(',','').strip()
            s = ''.join(filter(str.isdigit, s))
            s = s.zfill(8)  # Completa com zeros à esquerda para 8 dígitos
            return s
        df[col] = df[col].map(format_cep)
    return df
