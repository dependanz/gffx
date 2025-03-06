import math
import torch
from typing import Optional

class Transform:
    def __init__(
        self, 
        translation : Optional[list | torch.Tensor] = None,
        rotation    : Optional[list | torch.Tensor] = None,
        scale       : Optional[list | torch.Tensor] = None,
        device      : Optional[torch.device]        = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Defaults
        if not isinstance(translation, torch.Tensor) and not translation:
            translation = [0., 0., 0.]
        if isinstance(translation, list):
            translation = torch.tensor(translation, device=device)[None,:]
        
        if not isinstance(rotation, torch.Tensor) and not rotation:
            rotation = [0., 0., 0.]
        if isinstance(rotation, list):
            rotation = torch.tensor(rotation, device=device)[None,:]
            
        if not isinstance(scale, torch.Tensor) and not scale:
            scale = [1., 1., 1.]
        if isinstance(scale, list):
            scale = torch.tensor(scale, device=device)[None,:]
        
        # 
        self.translation = translation
        self.rotation    = rotation
        self.scale       = scale

class MeshObject:
    def __init__(
        self, 
        vertices : torch.Tensor,
        faces    : torch.Tensor,
        
        init_transform   : Optional[Transform]    = None,
        init_translation : Optional[torch.Tensor] = None,
        init_rotation    : Optional[torch.Tensor] = None,
        init_scale       : Optional[torch.Tensor] = None
    ):
        # Defaults
        if init_transform:
            assert init_translation is None, "If init_transform is not None, init_translation must be None"
            assert init_rotation    is None, "If init_transform is not None, init_rotation must be None"
            assert init_scale       is None, "If init_transform is not None, init_scale must be None"
            self.transform = init_transform
        else:
            self.transform = Transform(
                translation = init_translation,
                rotation    = init_rotation,
                scale       = init_scale
            )
        
        self.vertices = vertices
        self.faces    = faces
        
def generate_cube_mesh(
    init_transform   : Optional[Transform]           = None,
    init_translation : Optional[list | torch.Tensor] = None,
    init_rotation    : Optional[list | torch.Tensor] = None,
    init_scale       : Optional[list | torch.Tensor] = None,
    device           : Optional[torch.device]        = None
):
    # 
    if isinstance(init_translation, list):
        init_translation = torch.tensor(init_translation, device=device)
    if isinstance(init_rotation, list):
        init_rotation = torch.tensor(init_rotation, device=device)
    if isinstance(init_scale, list):
        init_scale = torch.tensor(init_scale, device=device)
    
    # Define vertices for a cube centered at the origin with side length 2
    vertices = torch.tensor([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ], device=device, dtype=torch.float32)

    # Define faces as two triangles per cube face (12 triangles total)
    faces = torch.tensor([
        [0, 1, 2], [1, 3, 2],  # Left face
        [4, 6, 5], [5, 6, 7],  # Right face
        [0, 4, 1], [1, 4, 5],  # Bottom face
        [2, 3, 6], [3, 7, 6],  # Top face
        [0, 2, 4], [2, 6, 4],  # Back face
        [1, 5, 3], [3, 5, 7]   # Front face
    ], device=device, dtype=torch.int64)

    return MeshObject(
        vertices         = vertices, 
        faces            = faces,
        init_transform   = init_transform,
        init_translation = init_translation,
        init_rotation    = init_rotation,
        init_scale       = init_scale
    )
    
    
def generate_icosphere_mesh(
    init_transform   : Optional[Transform]           = None,
    init_translation : Optional[list | torch.Tensor] = None,
    init_rotation    : Optional[list | torch.Tensor] = None,
    init_scale       : Optional[list | torch.Tensor] = None,
    device           : Optional[torch.device]        = None
):
    #
    if isinstance(init_translation, list):
        init_translation = torch.tensor(init_translation, device=device)
    if isinstance(init_rotation, list):
        init_rotation = torch.tensor(init_rotation, device=device)
    if isinstance(init_scale, list):
        init_scale = torch.tensor(init_scale, device=device)
    
    t = (1.0 + math.sqrt(5.0)) / 2.0

    verts = [
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ]
    
    # Normalize vertices to unit length
    vertices = []
    for v in verts:
        norm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        vertices.append([v[0] / norm, v[1] / norm, v[2] / norm])
    vertices = torch.tensor(vertices, device=device, dtype=torch.float32)

    faces = torch.tensor([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], device=device, dtype=torch.int64)

    return MeshObject(
        vertices         = vertices, 
        faces            = faces,
        init_transform   = init_transform,
        init_translation = init_translation,
        init_rotation    = init_rotation,
        init_scale       = init_scale
    )