"""Lazy image facade; arithmetic exists only in the selected backend implementation."""
__all__ = ["convert_alpha", "srgb_to_linear", "linear_to_srgb", "apply_mask",
           "composite", "reconstruct", "affine_matrix", "warp_affine", "adjust",
           "gaussian_blur", "capabilities"]


def convert_alpha(image, *, source="straight", target="premultiplied"):
    from .torch.image import convert_alpha as operation
    return operation(image, source=source, target=target)


def srgb_to_linear(rgb):
    from .torch.image import srgb_to_linear as operation
    return operation(rgb)


def linear_to_srgb(rgb):
    from .torch.image import linear_to_srgb as operation
    return operation(rgb)


def apply_mask(image, mask=None, *, opacity=1.0, alpha="premultiplied"):
    from .torch.image import apply_mask as operation
    return operation(image, mask, opacity=opacity, alpha=alpha)


def composite(layers, background, *, alpha="premultiplied"):
    from .torch.image import composite as operation
    return operation(layers, background, alpha=alpha)


def reconstruct(image, factor):
    from .torch.image import reconstruct as operation
    return operation(image, factor)


def affine_matrix(translation, rotation, scale, pivot):
    from .torch.image import affine_matrix as operation
    return operation(translation, rotation, scale, pivot)


def warp_affine(image, matrix, output_size, *, filter="bilinear", border="transparent", alpha="premultiplied"):
    from .torch.image import warp_affine as operation
    return operation(image, matrix, output_size, filter=filter, border=border, alpha=alpha)


def adjust(image, *, exposure=0.0, brightness=0.0, alpha="premultiplied"):
    from .torch.image import adjust as operation
    return operation(image, exposure=exposure, brightness=brightness, alpha=alpha)


def gaussian_blur(image, sigma, *, radius, border="transparent", alpha="premultiplied"):
    from .torch.image import gaussian_blur as operation
    return operation(image, sigma, radius=radius, border=border, alpha=alpha)


def capabilities():
    from .torch.image import capabilities as operation
    return operation()
