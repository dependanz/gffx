"""The full render chain, and the gradient that makes it differentiable.

These are the two figures the texture set does not contain. Everything before them shows a sampler;
these show a renderer, and then show that the renderer has a derivative.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

import gffx_ctypes as gx
from make_texture_figures import OUT, checkerboard, label, row, to_image


def uv_sphere(rings=24, sectors=48):
    """A UV sphere, generated here rather than loaded, so the figures carry no external asset.

    This matters beyond convenience: the strongest renders this project has produced are of a FLAME
    head, which is licensed for non-commercial research and must never be published. A procedural
    mesh is the honest replacement rather than a placeholder.
    """
    v, u = np.meshgrid(np.linspace(0, np.pi, rings), np.linspace(0, 2 * np.pi, sectors),
                       indexing="ij")
    positions = np.stack([np.sin(v) * np.cos(u), np.cos(v), np.sin(v) * np.sin(u)],
                         axis=-1).reshape(-1, 3)
    uvs = np.stack([u / (2 * np.pi), v / np.pi], axis=-1).reshape(-1, 2)
    faces = []
    for i in range(rings - 1):
        for j in range(sectors - 1):
            a = i * sectors + j
            # Counter-clockwise when seen from outside. The reversed winding renders an
            # identical silhouette, so the error is invisible until a normal is used: with
            # CULL_BACK it keeps the far hemisphere, whose object-space normals point away from
            # both camera and light. Measured that way it showed as n.l negative on 89 percent of
            # visible pixels while the picture still looked like a sphere.
            faces.append([a, a + 1, a + sectors])
            faces.append([a + 1, a + sectors + 1, a + sectors])
    return positions.astype(np.float64), np.array(faces, dtype=np.int32), uvs.astype(np.float64)


def look_at_projection(eye, target, up, fov_y, aspect, near=0.1, far=100.0):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    f = 1.0 / np.tan(fov_y * 0.5)
    projection = np.zeros((4, 4))
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2 * far * near) / (near - far)
    projection[3, 2] = -1.0
    return projection @ view


def figure_full_chain(session):
    """One textured render through five gffx operations, with nothing else in the pipeline.

    transforms.transform_points -> transforms.perspective_divide -> render.rasterize ->
    render.interpolate -> render.texture. The only arithmetic outside gffx is the camera matrix
    and a Lambert term on the interpolated normal, both of which are inputs to the library rather
    than substitutes for it.
    """
    positions, faces, uvs = uv_sphere()
    height, width = 320, 320
    matrix = look_at_projection(np.array([2.4, 1.5, 2.4]), np.zeros(3),
                                np.array([0.0, 1.0, 0.0]), np.radians(40.0), width / height)
    offsets = np.array([0, positions.shape[0]], dtype=np.int32)
    face_offsets = np.array([0, faces.shape[0]], dtype=np.int32)

    homogeneous = session.transform_points(positions, matrix.reshape(1, 4, 4), offsets)
    ndc, _ = session.perspective_divide(homogeneous)
    face_index, barycentric, _, _ = session.rasterize(
        ndc, faces, offsets, face_offsets, height, width, cull_mode=gx.CULL_BACK)

    # Per-corner attributes: the texture coordinate and the object-space normal, which for a unit
    # sphere is the position. Both are interpolated by the same call.
    corner = np.concatenate([uvs[faces], positions[faces]], axis=2)
    interpolated = session.interpolate(face_index, barycentric, corner)[0, :, :, 0, :]

    coords = np.ascontiguousarray(interpolated[:, :, :2].reshape(-1, 2))
    texture = checkerboard(256, 16)
    pyramid, pyramid_offsets = session.texture_pyramid(texture, 0)
    dudx = np.gradient(interpolated[:, :, 0], axis=1)
    dvdx = np.gradient(interpolated[:, :, 1], axis=1)
    dudy = np.gradient(interpolated[:, :, 0], axis=0)
    dvdy = np.gradient(interpolated[:, :, 1], axis=0)
    derivatives = np.ascontiguousarray(
        np.stack([dudx.ravel(), dvdx.ravel(), dudy.ravel(), dvdy.ravel()], axis=1))
    albedo = session.texture(pyramid, pyramid_offsets, 256, 256, coords, 3,
                             filter=gx.FILTER_BILINEAR, mip_filter=gx.MIP_LINEAR,
                             derivatives=derivatives).reshape(height, width, 3)

    normal = interpolated[:, :, 2:5]
    length = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = np.divide(normal, np.maximum(length, 1e-9))
    light = np.array([0.5, 0.7, 0.6])
    light = light / np.linalg.norm(light)
    lambert = np.clip((normal * light).sum(axis=2), 0.0, 1.0)[:, :, None]

    background = face_index[0, :, :, 0] < 0
    shaded = albedo * (0.25 + 0.75 * lambert)
    shaded[background] = 0.97

    # Every panel gets the same pale background, so the eye compares content rather than framing.
    def on_background(rgb):
        return rgb * ~background[:, :, None] + 0.97 * background[:, :, None]

    coverage = np.repeat((~background)[:, :, None].astype(np.float64), 3, axis=2) * 0.35 + 0.2
    uv_image = np.concatenate([interpolated[:, :, :2], np.zeros((height, width, 1))], axis=2)
    panels = [
        label(to_image(on_background(coverage)), "render.rasterize coverage"),
        label(to_image(on_background(uv_image)), "render.interpolate UVs"),
        label(to_image(on_background(albedo)), "render.texture albedo"),
        label(to_image(shaded), "Lambert on the interpolated normal"),
    ]
    row(panels).save(os.path.join(OUT, "06-full-chain.png"))
    return "06-full-chain.png"


def figure_gradient(session):
    """Gradient descent on vertex positions, driven entirely by gffx's own backward passes.

    An arrow diagram was the first attempt and was weak: the gradient minimises a difference of
    distance fields, so the arrows do not point at the target's vertices and a reader cannot tell
    from them whether the gradient is useful or merely nonzero. Running the optimisation answers
    that directly - either the shape converges or it does not.

    A note kept from an earlier attempt, because it is the more interesting fact. Driving the same
    loss through an interpolated constant attribute under hard rasterization gave gradients of
    order 1e-14, which is zero: the interior value does not change as a vertex moves and coverage
    changes discontinuously. signed_distance is the continuous quantity, and it carries a usable
    derivative even at blur_radius_px of zero. What a nonzero blur buys is gradient from a wider
    band of pixels, not the existence of a gradient.
    """
    height, width = 200, 200
    ndc = np.array([[-0.55, -0.45, 0.0], [0.60, -0.35, 0.0], [-0.05, 0.62, 0.0]])
    target_ndc = np.array([[-0.20, -0.62, 0.0], [0.70, 0.05, 0.0], [-0.40, 0.30, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    offsets = np.array([0, 3], dtype=np.int32)
    face_offsets = np.array([0, 1], dtype=np.int32)

    def render(vertices):
        return session.rasterize(vertices, faces, offsets, face_offsets, height, width,
                                 blur_radius_px=6.0, cull_mode=gx.CULL_NONE)

    _, _, _, target_distance = render(target_ndc)
    target_inside = target_distance[0, :, :, 0] < 0.0

    def step(vertices):
        index, bary, _, distance = render(vertices)
        covered = (index[0, :, :, 0] >= 0) & (target_distance[0, :, :, 0] < 1e30)
        # Subtract only on covered pixels. Computing the difference over the whole image first
        # forms inf minus inf on background pixels, which is a NaN that the mask then discards -
        # correct, but it emits a warning and relies on the mask to clean up after itself.
        residual = np.zeros_like(distance)
        residual[0, :, :, 0][covered] = (distance[0, :, :, 0][covered]
                                         - target_distance[0, :, :, 0][covered])
        grad = session.rasterize_backward(vertices, faces, height, width, index,
                                          np.zeros_like(bary), np.zeros_like(distance), residual)
        inside = distance[0, :, :, 0] < 0.0
        union = float((inside | target_inside).sum())
        iou = float((inside & target_inside).sum()) / max(union, 1.0)
        return grad, inside, iou

    # The step is normalised by the peak gradient rather than being a raw learning rate, because
    # signed_distance is in squared pixel units and its scale is a property of the image size
    # rather than of the problem.
    frames = []
    shots = (0, 4, 12, 40, 120)
    current = ndc.copy()
    for iteration in range(shots[-1] + 1):
        grad, inside, iou = step(current)
        if iteration in shots:
            canvas = np.full((height, width, 3), 0.97)
            canvas[target_inside] = np.array([0.82, 0.88, 0.80])
            canvas[inside] = np.array([0.30, 0.42, 0.62])
            canvas[inside & target_inside] = np.array([0.36, 0.60, 0.52])
            frames.append(label(to_image(canvas).resize((width, height), Image.NEAREST),
                                "step %d   IoU %.3f" % (iteration, iou)))
        peak = np.abs(grad).max()
        if peak > 1e-12:
            current = current - (0.02 / peak) * grad
    row(frames).save(os.path.join(OUT, "07-gradient-descent.png"))
    final_iou = step(current)[2]
    return "07-gradient-descent.png", final_iou


if __name__ == "__main__":
    session = gx.Session()
    print("wrote", figure_full_chain(session))
    name, iou = figure_gradient(session)
    print("wrote", name, "final IoU %.4f" % iou)
