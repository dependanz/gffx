"""Figures for the render.texture pipeline, rendered by calling gffx through its C ABI.

Every pixel in every figure here comes from gffx_render_texture. Nothing is drawn by matplotlib or
PIL beyond arranging the finished images and labelling them, because a figure that showcases a
library has to be produced by the library or it showcases the plotting code instead.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

import gffx_ctypes as gx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)


def checkerboard(size=256, squares=16, rgb=True):
    """A checkerboard is the standard minification test because its high frequency is exactly what
    aliases; a smooth texture would look acceptable however badly it were sampled."""
    step = size // squares
    grid = ((np.arange(size)[:, None] // step + np.arange(size)[None, :] // step) % 2)
    if not rgb:
        return grid.astype(np.float64)[:, :, None]
    image = np.zeros((size, size, 3), dtype=np.float64)
    warm = np.array([0.92, 0.35, 0.18])
    cool = np.array([0.10, 0.22, 0.38])
    image[grid == 1] = warm
    image[grid == 0] = cool
    return image


def to_image(array):
    return Image.fromarray(np.clip(array, 0.0, 1.0).astype(np.float32).__mul__(255.0)
                           .astype(np.uint8))


def label(image, text, height=22):
    """Captions go under the image rather than over it, so no sample is hidden by a label.

    The panel widens to fit its caption when the image is narrower than the text. Without that, a
    small panel's caption runs into the next panel and the figure reads as though the caption
    belongs to its neighbour.
    """
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_width = int(draw.textlength(text)) + 8
    width = max(image.width, text_width)
    out = Image.new("RGB", (width, image.height + height), (255, 255, 255))
    out.paste(image, ((width - image.width) // 2, 0))
    draw = ImageDraw.Draw(out)
    draw.text(((width - text_width) // 2 + 4, image.height + 5), text, fill=(20, 20, 20))
    return out


def row(images, gap=8):
    width = sum(i.width for i in images) + gap * (len(images) - 1)
    height = max(i.height for i in images)
    out = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width + gap
    return out


def sample_grid(session, pyramid, offsets, h, w, channels, us, vs, **kwargs):
    coords = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)
    samples = session.texture(pyramid, offsets, h, w, coords, channels, **kwargs)
    return samples.reshape(us.shape + (channels,))


def figure_filtering(session):
    """NEAREST against BILINEAR, magnified hard enough that the difference is the whole point."""
    texture = checkerboard(16, 4)
    pyramid, offsets = session.texture_pyramid(texture, 1)
    size = 256
    axis = (np.arange(size) + 0.5) / size * 0.5 + 0.25
    us, vs = np.meshgrid(axis, axis)
    panels = []
    for name, mode in (("NEAREST", gx.FILTER_NEAREST), ("BILINEAR", gx.FILTER_BILINEAR)):
        image = sample_grid(session, pyramid, offsets, 16, 16, 3, us, vs, filter=mode,
                            wrap_u=gx.WRAP_CLAMP, wrap_v=gx.WRAP_CLAMP)
        panels.append(label(to_image(image), "filter = %s" % name))
    row(panels).save(os.path.join(OUT, "01-filtering.png"))
    return "01-filtering.png"


def figure_wrap_modes(session):
    """All four wrap modes over the same coordinates, which run well outside [0,1] on both axes so
    every mode has to resolve rather than only the clamped one."""
    texture = checkerboard(64, 4)
    texture[:8, :, :] = np.array([0.95, 0.85, 0.20])   # a marked edge, so mirroring is visible
    pyramid, offsets = session.texture_pyramid(texture, 1)
    size = 192
    axis = (np.arange(size) + 0.5) / size * 3.0 - 1.0
    us, vs = np.meshgrid(axis, axis)
    border = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    panels = []
    for name, mode in (("REPEAT", gx.WRAP_REPEAT), ("CLAMP", gx.WRAP_CLAMP),
                       ("MIRROR", gx.WRAP_MIRROR), ("BORDER", gx.WRAP_BORDER)):
        image = sample_grid(session, pyramid, offsets, 64, 64, 3, us, vs,
                            filter=gx.FILTER_BILINEAR, wrap_u=mode, wrap_v=mode, border=border)
        panels.append(label(to_image(image), "wrap = %s" % name))
    row(panels).save(os.path.join(OUT, "02-wrap-modes.png"))
    return "02-wrap-modes.png"


def figure_pyramid(session):
    """The level chain at native size, so the halving is literal rather than described.

    Drawn at a common size instead, the first several levels look identical: a checkerboard
    survives being halved until the level can no longer represent its frequency, at which point it
    collapses to the mean in one step. That reads as nothing happening and then everything
    happening, which is the opposite of what the pyramid is doing. Native size shows the chain
    actually shrinking, and a fine checker makes the averaging visible on the way down.
    """
    texture = checkerboard(128, 32)
    pyramid, offsets = session.texture_pyramid(texture, 0)
    panels = []
    h, w = 128, 128
    for level in range(len(offsets) - 1):
        block = pyramid[offsets[level]:offsets[level + 1]].reshape(h, w, 3)
        panels.append(label(to_image(block), "level %d  %dx%d" % (level, h, w)))
        h = max(1, h // 2)
        w = max(1, w // 2)
    row(panels).save(os.path.join(OUT, "03-mip-pyramid.png"))
    return "03-mip-pyramid.png"


def figure_minification(session):
    """A ground plane receding to the horizon: the case mipmapping exists for.

    The camera geometry is computed here and the texture coordinates handed to gffx, along with
    their screen-space derivatives. gffx chooses the level from those derivatives; the caller never
    picks a level. The three panels differ only in what the sampler is allowed to use, so any
    difference between them is the sampler's.
    """
    texture = checkerboard(256, 32)
    pyramid, offsets = session.texture_pyramid(texture, 0)
    width, height = 480, 240
    x = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    y = (np.arange(height) + 0.5) / height
    xx, yy = np.meshgrid(x, y)
    # A plane under a camera at unit height: depth falls as 1/y, which is what makes the texture
    # frequency rise without bound towards the horizon.
    depth = 1.0 / np.maximum(yy, 1e-4)
    us = xx * depth * 0.5
    vs = depth * 0.25
    # Screen-space derivatives by differencing neighbours. The contract is explicit that the
    # operation never derives these itself, because that would make each sample depend on its
    # neighbours; supplying them is the caller's job and this is what that looks like.
    dudx = np.gradient(us, axis=1)
    dvdx = np.gradient(vs, axis=1)
    dudy = np.gradient(us, axis=0)
    dvdy = np.gradient(vs, axis=0)
    derivatives = np.stack([dudx.ravel(), dvdx.ravel(), dudy.ravel(), dvdy.ravel()],
                           axis=1).astype(np.float64)
    coords = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)

    panels = []
    cases = (
        ("no mipmap (level 0 only)", None, gx.MIP_NEAREST),
        ("mip NEAREST", derivatives, gx.MIP_NEAREST),
        ("mip LINEAR (trilinear)", derivatives, gx.MIP_LINEAR),
    )
    for name, derivative, mip in cases:
        samples = session.texture(pyramid, offsets, 256, 256, coords, 3,
                                  filter=gx.FILTER_BILINEAR, mip_filter=mip,
                                  wrap_u=gx.WRAP_REPEAT, wrap_v=gx.WRAP_REPEAT,
                                  derivatives=derivative)
        panels.append(label(to_image(samples.reshape(height, width, 3)), name))
    row(panels).save(os.path.join(OUT, "04-minification.png"))
    return "04-minification.png"


def figure_lod(session):
    """The level gffx selects, drawn directly, so the choice is visible rather than inferred."""
    texture = checkerboard(256, 32)
    pyramid, offsets = session.texture_pyramid(texture, 0)
    levels = len(offsets) - 1
    width, height = 480, 240
    x = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    y = (np.arange(height) + 0.5) / height
    xx, yy = np.meshgrid(x, y)
    depth = 1.0 / np.maximum(yy, 1e-4)
    us, vs = xx * depth * 0.5, depth * 0.25
    dudx, dvdx = np.gradient(us, axis=1), np.gradient(vs, axis=1)
    dudy, dvdy = np.gradient(us, axis=0), np.gradient(vs, axis=0)
    derivatives = np.stack([dudx.ravel(), dvdx.ravel(), dudy.ravel(), dvdy.ravel()],
                           axis=1).astype(np.float64)
    coords = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)
    # Each level is filled with its own index, so sampling returns the level that was read. The
    # probe needs its own single-channel pyramid: reusing the three-channel offsets while claiming
    # one channel is rejected, correctly, by the sampler cross-checking offsets against extents.
    probe_texture = np.zeros((256, 256, 1), dtype=np.float64)
    probe, probe_offsets = session.texture_pyramid(probe_texture, 0)
    for level in range(levels):
        probe[probe_offsets[level]:probe_offsets[level + 1]] = float(level)
    chosen = session.texture(probe, probe_offsets, 256, 256, coords, 1,
                             filter=gx.FILTER_NEAREST, mip_filter=gx.MIP_NEAREST,
                             derivatives=derivatives).reshape(height, width)
    shaded = np.zeros((height, width, 3))
    palette = np.array([[0.16, 0.20, 0.42], [0.20, 0.45, 0.60], [0.35, 0.68, 0.60],
                        [0.75, 0.80, 0.42], [0.93, 0.66, 0.30], [0.88, 0.40, 0.32],
                        [0.72, 0.25, 0.38], [0.45, 0.16, 0.34], [0.25, 0.10, 0.22]])
    for level in range(levels):
        shaded[np.round(chosen).astype(int) == level] = palette[level % len(palette)]
    label(to_image(shaded), "level of detail chosen by gffx from caller-supplied derivatives"
          ).save(os.path.join(OUT, "05-lod-selection.png"))
    return "05-lod-selection.png"


if __name__ == "__main__":
    session = gx.Session()
    for maker in (figure_filtering, figure_wrap_modes, figure_pyramid, figure_minification,
                  figure_lod):
        name = maker(session)
        print("wrote", name)
