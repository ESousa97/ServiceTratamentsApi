# Exemplo simples de limitador de memória decorador
import functools
import psutil

def memory_limiter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            raise MemoryError("Limite de memória excedido!")
        return func(*args, **kwargs)
    return wrapper
