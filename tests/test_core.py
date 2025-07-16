import pandas as pd
from core.file_handler import FileHandler

def test_load_from_bytes():
    csv_content = b"id,nome\n1,Ana\n2,Bruno"
    df = FileHandler.load_from_bytes(csv_content)
    assert df.shape == (2, 2)
