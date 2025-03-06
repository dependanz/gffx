import math
import gffx
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
        self.translation = translation.float()
        self.rotation    = rotation.float()
        self.scale       = scale.float()

class MeshObject:
    def __init__(
        self, 
        vertices         : torch.Tensor,
        faces            : torch.Tensor,
        normals          : Optional[torch.Tensor] = None,
        init_transform   : Optional[Transform]    = None,
        init_translation : Optional[torch.Tensor] = None,
        init_rotation    : Optional[torch.Tensor] = None,
        init_scale       : Optional[torch.Tensor] = None,
        
        ambient_color    : list | torch.Tensor = [0.5, 0.5, 0.5],
        diffuse_color    : list | torch.Tensor = [0.5, 0.5, 0.5],
        specular_color   : list | torch.Tensor = [0.5, 0.5, 0.5],
        specular_coefficient : float = 1.0
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
        self.normals  = normals
        self.device   = vertices.device
        
        # [4NOW] Flat colors
        self.ambient_color = ambient_color
        if isinstance(self.ambient_color, list):
            self.ambient_color = torch.tensor(self.ambient_color, device=self.device, dtype=torch.float32)
            
        self.diffuse_color = diffuse_color
        if isinstance(self.diffuse_color, list):
            self.diffuse_color = torch.tensor(self.diffuse_color, device=self.device, dtype=torch.float32)
            
        self.specular_color = specular_color
        if isinstance(self.specular_color, list):
            self.specular_color = torch.tensor(self.specular_color, device=self.device, dtype=torch.float32)
        
        self.specular_coefficient = specular_coefficient
        
    def get_transformed(self):
        """
            Apply transformation to vertices
            
            Returns:
            -------
            torch.Tensor: Transformed vertices
            
            if normals are not None, also returns transformed normals
        """
        M = gffx.linalg.transformation_matrix(
            translation_vec = self.transform.translation[None,...],
            rotation_vec    = self.transform.rotation[None,...],
            scale_vec       = self.transform.scale[None,...]
        ) # dim(1, 4, 4)
        vertices_h = torch.cat([self.vertices, torch.ones((self.vertices.shape[0], 1), device=self.device)], dim=-1)
        vertices_h = vertices_h @ M.transpose(-1, -2)
        
        # [Optional] Transform normals
        if self.normals is not None:
            M = M[:, 0:3, 0:3].inverse()
            normals = self.normals @ M
            return vertices_h[0,..., 0:3], normals[0]
        
        return vertices_h[0,..., 0:3], None
    
def compute_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
        Compute per-vertex normals for a batch of meshes.

        Args:
            vertices (torch.Tensor): Tensor of shape (B, V, 3) with vertex positions.
            faces (torch.Tensor): Long tensor of shape (B, F, 3) with indices into vertices.

        Returns:
            torch.Tensor: Tensor of shape (B, V, 3) with normalized vertex normals.
    """
    B, V, _ = vertices.shape
    _, F, _ = faces.shape
    device = vertices.device

    # Initialize normals accumulator
    vertex_normals = torch.zeros_like(vertices)

    # Gather vertex positions for each corner of every face
    v0 = torch.gather(vertices, 1, faces[:, :, 0].unsqueeze(-1).expand(-1, -1, 3))
    v1 = torch.gather(vertices, 1, faces[:, :, 1].unsqueeze(-1).expand(-1, -1, 3))
    v2 = torch.gather(vertices, 1, faces[:, :, 2].unsqueeze(-1).expand(-1, -1, 3))

    # Compute face normals (unnormalized) via cross product
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=2) # dim(B,F,3)
    face_normals = face_normals / (face_normals.norm(dim=2, keepdim=True) + 1e-8)

    # Accumulate face normals to each vertex using scatter_add_
    for i in range(3):
        idx = faces[:, :, i].unsqueeze(-1).expand(-1, -1, 3)
        vertex_normals.scatter_add_(1, idx, face_normals)

    # Normalize the accumulated vertex normals
    norm = vertex_normals.norm(dim=2, keepdim=True).clamp(min=1e-8)
    vertex_normals = vertex_normals / norm

    return vertex_normals

def mesh_from_vertices_and_faces(
    vertices         : torch.Tensor,
    faces            : torch.Tensor,
    init_transform   : Optional[Transform]           = None,
    init_translation : Optional[list | torch.Tensor] = None,
    init_rotation    : Optional[list | torch.Tensor] = None,
    init_scale       : Optional[list | torch.Tensor] = None,
    
    ambient_color  : list | torch.Tensor = [0.5, 0.5, 0.5],
    diffuse_color  : list | torch.Tensor = [0.5, 0.5, 0.5],
    specular_color : list | torch.Tensor = [0.5, 0.5, 0.5],
    specular_coefficient : float = 1.0,
    
    device           : Optional[torch.device]        = None
):
    # 
    if isinstance(init_translation, list):
        init_translation = torch.tensor(init_translation, device=device)
    if isinstance(init_rotation, list):
        init_rotation = torch.tensor(init_rotation, device=device)
    if isinstance(init_scale, list):
        init_scale = torch.tensor(init_scale, device=device)
    
    # Compute Normals
    normals = compute_normals(vertices[None,:], faces[None,:])[0]
    
    return MeshObject(
        vertices         = vertices,
        faces            = faces,
        normals          = normals,
        init_transform   = init_transform,
        init_translation = init_translation,
        init_rotation    = init_rotation,
        init_scale       = init_scale,
        
        ambient_color  = ambient_color,
        diffuse_color  = diffuse_color,
        specular_color = specular_color,
        specular_coefficient = specular_coefficient
    )
    
def generate_cube_mesh(
    init_transform   : Optional[Transform]           = None,
    init_translation : Optional[list | torch.Tensor] = None,
    init_rotation    : Optional[list | torch.Tensor] = None,
    init_scale       : Optional[list | torch.Tensor] = None,
    
    ambient_color  : list | torch.Tensor = [0.5, 0.5, 0.5],
    diffuse_color  : list | torch.Tensor = [0.5, 0.5, 0.5],
    specular_color : list | torch.Tensor = [0.5, 0.5, 0.5],
    specular_coefficient : float = 1.0,
    
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
        normals          = vertices / torch.linalg.norm(vertices, dim=-1, keepdim=True),
        init_transform   = init_transform,
        init_translation = init_translation,
        init_rotation    = init_rotation,
        init_scale       = init_scale,
        
        ambient_color  = ambient_color,
        diffuse_color  = diffuse_color,
        specular_color = specular_color,
        specular_coefficient = specular_coefficient
    )
    
    
def generate_icosphere_mesh(
    num_subdivisions : int                           = 1,
    init_transform   : Optional[Transform]           = None,
    init_translation : Optional[list | torch.Tensor] = None,
    init_rotation    : Optional[list | torch.Tensor] = None,
    init_scale       : Optional[list | torch.Tensor] = None,
    
    ambient_color  : list | torch.Tensor = [0.5, 0.5, 0.5],
    diffuse_color  : list | torch.Tensor = [0.5, 0.5, 0.5],
    specular_color : list | torch.Tensor = [0.5, 0.5, 0.5],
    specular_coefficient : float = 1.0,
    
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

    # Subdivide faces
    for _ in range(num_subdivisions):
        new_faces = []
        midpoints = {}
        for face in faces:
            a, b, c = face.tolist()
            ab = tuple(sorted((a, b)))
            ac = tuple(sorted((a, c)))
            bc = tuple(sorted((b, c)))

            if ab not in midpoints:
                midpoints[ab] = len(vertices)
                vertices = torch.cat([vertices, (vertices[a:a+1] + vertices[b:b+1]) / 2.0], dim=0)
            if ac not in midpoints:
                midpoints[ac] = len(vertices)
                vertices = torch.cat([vertices, (vertices[a:a+1] + vertices[c:c+1]) / 2.0], dim=0)
            if bc not in midpoints:
                midpoints[bc] = len(vertices)
                vertices = torch.cat([vertices, (vertices[b:b+1] + vertices[c:c+1]) / 2.0], dim=0)

            d = midpoints[ab]
            e = midpoints[ac]
            f = midpoints[bc]

            new_faces.extend([
                [a, d, e],
                [b, d, f],
                [c, e, f],
                [d, e, f]
            ])
        faces = torch.tensor(new_faces, device=device, dtype=torch.int64)
    
    # Project vertices to unit length
    vertices = torch.nn.functional.normalize(vertices, dim=-1)
    vertices = torch.cat([vertices, torch.ones((vertices.shape[0], 1), device=device)], dim=-1)
    vertices = vertices @ torch.diag(torch.tensor([1, 1, 1, 0], device=device)).to(vertices.dtype)
    vertices = vertices[..., 0:3]
    
    return MeshObject(
        vertices         = vertices,
        faces            = faces,
        normals          = vertices,
        init_transform   = init_transform,
        init_translation = init_translation,
        init_rotation    = init_rotation,
        init_scale       = init_scale,
        
        ambient_color  = ambient_color,
        diffuse_color  = diffuse_color,
        specular_color = specular_color,
        specular_coefficient = specular_coefficient
    )