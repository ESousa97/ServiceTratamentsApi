import os

def validate_file(file_path, max_rows, max_size_mb):
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    if size_mb > max_size_mb:
        raise ValueError("Arquivo excede o tamanho máximo permitido.")
    return True
