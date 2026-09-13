"""Kinetic Bloom: finite native rig deformation, not physics or a beauty oracle.

All moving surface vertices pass through public blend_shapes, Rodrigues, FK and
LBS with inverse bind. Embedded markers use evaluate_barycentric. Display normals
and the inherited studio-lighting recipe are presentation only.
"""
import math

import torch

FRAMES = 24
JOINTS = 5


def fixture():
    vertices, faces, colors = [], [], []
    rings, sides = 32, 12
    for part in (-1, 0, 1):
        start = len(vertices)
        for row in range(rings + 1):
            u = row / rings
            y = .35 + 2.7 * u
            for column in range(sides):
                angle = math.tau * column / sides + (0 if part == 0 else part * .7 * u)
                if part == 0:
                    x, z = .055 * math.cos(angle), .055 * math.sin(angle)
                    color = [.78, .46, .14]
                else:
                    radius = .035 + .43 * math.sin(math.pi * u) ** 1.3
                    x = part * (.13 + .29 * math.sin(math.pi * u)) + radius * math.cos(angle)
                    z = part * .035 + .25 * radius * math.sin(angle)
                    cold, warm = [.025, .45, .48], [.95, .24, .075]
                    color = [component * (.68 + .32 * u) for component in (cold if part < 0 else warm)]
                    if column in (0, sides // 2):
                        color = [.9, .65, .27]
                vertices.append([x, y, z])
                colors.append(color)
        for row in range(rings):
            for column in range(sides):
                a = start + row * sides + column
                b = start + row * sides + (column + 1) % sides
                faces.extend([[a, a + sides, b], [b, a + sides, b + sides]])
        faces.extend([[start, start + (column + 1) % sides, start + column] for column in range(1, sides - 1)])
        end = start + rings * sides
        faces.extend([[end, end + column, end + (column + 1) % sides] for column in range(1, sides - 1)])
    template = torch.tensor(vertices, dtype=torch.float32)
    topology = torch.tensor(faces, dtype=torch.int32)
    colors = torch.tensor(colors, dtype=template.dtype)
    u = ((template[:, 1] - .35) / 2.7).clamp(0, 1)
    skin_weights = torch.zeros((len(template), JOINTS), dtype=template.dtype)
    lower = (u * (JOINTS - 1)).floor().long().clamp(max=JOINTS - 2)
    blend = u * (JOINTS - 1) - lower
    indices = torch.arange(len(template))
    skin_weights[indices, lower] = 1 - blend
    skin_weights[indices, lower + 1] = blend
    deltas = torch.zeros((2, len(template), 3), dtype=template.dtype)
    deltas[0, :, 0] = template[:, 0] * torch.sin(math.pi * u) * .25
    deltas[1, :, 2] = torch.sin(math.pi * u) * .10
    translations = torch.zeros((1, JOINTS, 3, 1), dtype=template.dtype)
    translations[0, 0, 1, 0] = .35
    translations[0, 1:, 1, 0] = 2.7 / (JOINTS - 1)
    inverse_bind = torch.eye(4, dtype=template.dtype).repeat(1, JOINTS, 1, 1)
    inverse_bind[0, :, 1, 3] = -(.35 + torch.arange(JOINTS) * 2.7 / (JOINTS - 1))
    return dict(template=template, faces=topology, colors=colors, deltas=deltas,
                skin_weights=skin_weights, translations=translations, inverse_bind=inverse_bind,
                parents=torch.arange(-1, JOINTS - 1, dtype=torch.int32))


def normals(vertices, faces):
    triangle = vertices[faces.long()]
    normal = torch.linalg.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    result = torch.zeros_like(vertices)
    for corner in range(3):
        result.index_add_(0, faces[:, corner].long(), normal)
    return torch.nn.functional.normalize(result, dim=-1)


@torch.no_grad()
def frame(phase, data=None):
    import gffx
    from .scene_showcase import cylinder
    data = fixture() if data is None else data
    adapter = gffx.torch
    coefficients = data['template'].new_tensor([[math.sin(phase), math.cos(phase) * .4]])
    posed = adapter.mesh.blend_shapes(data['template'], data['deltas'], coefficients)
    pose = torch.zeros((1, JOINTS, 3), dtype=posed.dtype)
    for joint in range(JOINTS):
        pose[0, joint, 2] = .13 * math.sin(phase - joint * .50)
        pose[0, joint, 1] = .16 * math.cos(phase + joint * .35)
        pose[0, joint, 0] = .07 * math.sin(phase + joint * .40)
    rotations = adapter.transforms.rodrigues(pose)
    last = posed.new_tensor([0., 0., 0., 1.]).reshape(1, 1, 1, 4).repeat(1, JOINTS, 1, 1)
    local = torch.cat((torch.cat((rotations, data['translations']), -1), last), -2)
    world = adapter.skeleton.forward_kinematics(local, data['parents'])
    skinning = world @ data['inverse_bind']
    moving = adapter.mesh.skin_lbs(posed, skinning, data['skin_weights'])[0]
    selected = torch.tensor([50, 700, 1500], dtype=torch.int32)
    bary = moving.new_tensor([[.25, .25, .5]]).repeat(3, 1)
    markers = adapter.mesh.evaluate_barycentric(moving[None], data['faces'], selected, bary)[0]
    # Independent tensor expressions quantify the displayed deformation, rather
    # than treating a pleasant image as proof that the native rig is correct.
    expected_posed = data['template'] + torch.einsum('bk,kvc->bvc', coefficients, data['deltas'])[0]
    homogeneous = torch.cat((expected_posed, torch.ones_like(expected_posed[:, :1])), -1)
    expected = torch.einsum('vj,jca,va->vc', data['skin_weights'], skinning[0, :, :3], homogeneous)
    error = float((moving - expected).abs().max())
    if not math.isfinite(error) or error > 2e-6:
        raise ValueError('articulated display differs from its independent skinning expression')
    base, base_faces, base_normals = cylinder(1.05, .26, segments=40)
    top, top_faces, top_normals = cylinder(.98, .035, segments=40)
    top[:, 1] += .26
    vertices = torch.cat((moving, base, top))
    faces = torch.cat((data['faces'], base_faces + len(moving), top_faces + len(moving) + len(base)))
    return dict(vertices=vertices, faces=faces,
        normals=torch.cat((normals(moving, data['faces']), base_normals, top_normals)),
        colors=torch.cat((data['colors'], moving.new_tensor([[.055, .08, .10]]).expand(len(base), 3),
                          moving.new_tensor([[.73, .46, .17]]).expand(len(top), 3))),
        markers=markers, max_native_reference_error=error, joint_positions=world[0, :, :3, 3])


def payload(check=lambda: None, *, diagnostic=False):
    from .scene_showcase import beauty
    data = fixture()
    clip = []
    for index in range(FRAMES):
        check()
        clip.append(frame(math.tau * index / FRAMES, data))
    images = []
    for index in (0, 6, 12):
        check()
        images.append(beauty(clip[index], eye=(3.6, 3.0, 5.4), size=(180, 320) if diagnostic else (720, 1280)))
    return dict(clip=clip, images=images,
                maximum_error=max(row['max_native_reference_error'] for row in clip))
