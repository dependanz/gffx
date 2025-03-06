class InteractiveDev:
    def __init__(
        self,
        enabled : bool = False
    ):
        self.enabled = enabled
    
    def __enter__(self):
        breakpoint()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        breakpoint()
        pass