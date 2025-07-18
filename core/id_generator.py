# core/id_generator.py
import uuid

def ensure_id_column(df):
    if 'id' not in df.columns:
        df.insert(0, 'id', [str(uuid.uuid4()) for _ in range(len(df))])
    return df
