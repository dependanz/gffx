"""Lazy material facade; currently binds explicit tensors to the PyTorch adapter."""
__all__ = ['classic', 'principled', 'custom']


def classic(normals, view, directions, *, albedo, specular, exponent, two_sided=False):
    from .torch.material import classic as operation
    return operation(normals, view, directions, albedo=albedo, specular=specular,
                     exponent=exponent, two_sided=two_sided)


def principled(normals, view, directions, *, albedo, roughness, f0, budget,
               two_sided=False, workspace_limit_bytes=67108864,
               allow_experimental_budget=False):
    from .torch import material
    return material.principled(normals, view, directions, albedo=albedo,
        roughness=roughness, f0=f0, budget=budget, two_sided=two_sided,
        workspace_limit_bytes=workspace_limit_bytes,
        allow_experimental_budget=allow_experimental_budget)


def custom(function, normals, view, directions, *, parameters, two_sided=False):
    from .torch import material
    return material.custom(function, normals, view, directions,
                           parameters=parameters, two_sided=two_sided)
