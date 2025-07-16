import pandas as pd

def validate_csv(df: pd.DataFrame) -> bool:
    if df.empty:
        raise ValueError("Arquivo CSV vazio!")
    if len(df.columns) < 1:
        raise ValueError("Nenhuma coluna detectada!")
    return True
