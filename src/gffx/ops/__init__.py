try:
    from ._cuda import add_vectors
    CUDA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    from ._cpu_stub import add_vectors
    CUDA_AVAILABLE = False