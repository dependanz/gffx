"""Eager camera facade. The sole currently implemented adapter is PyTorch.

This re-export performs no arithmetic: construction binds the shared composition
to PyTorch. Additional adapters must add explicit tensor-type dispatch here.
"""
from .torch.camera import Camera, ProjectedPoints

__all__=['Camera','ProjectedPoints']
