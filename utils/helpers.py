import pandas as pd
from io import BytesIO

def read_csv_from_bytes(file_bytes):
    return pd.read_csv(BytesIO(file_bytes))

def to_json_response(data):
    import json
    return json.dumps(data, ensure_ascii=False)
