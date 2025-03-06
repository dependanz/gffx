import gffx
import torch
from tqdm import tqdm

from .camera import *

def ray_sphere_intersection(ray_origins, ray_directions, sphere_pos, sphere_radius):
    # Check discriminant (ray-sphere intersection)
    e_min_c = ray_origins - sphere_pos[None, None, None, :]                    # dim(B, H, W, 3)
    d_dot_e_min_c = torch.sum(ray_directions * e_min_c, dim=-1, keepdim=True)  # dim(B, H, W, 1)
    d_dot_d = torch.sum(ray_directions * ray_directions, dim=-1, keepdim=True) # dim(B, H, W, 1)
    discriminant = d_dot_e_min_c ** 2 - d_dot_d * (torch.sum(e_min_c * e_min_c, dim=-1, keepdim=True) - sphere_radius ** 2)

    # Compute t
    hit_mask = (discriminant >= 0).float()
    t = torch.sqrt(discriminant * hit_mask) / d_dot_d
    t_minus = torch.clip(-d_dot_e_min_c - t, 0)
    t_plus  = torch.clip(-d_dot_e_min_c + t, 0)
    t = torch.maximum(t_minus, t_plus) * hit_mask

    return t

def ray_triangle_intersection(
    ray_origins, 
    ray_directions, 
    triangle_vertices, 
    t0 = 0, 
    t1 = 100
):
    """
        Compute ray-triangle intersection using Cramer's Rule
        
        Terms
        -----
        B = batch size
        N = number of triangles
        
        Parameters
        ----------
        ray_origins : torch.Tensor
            Ray origins -> dim(B, 3)
        ray_directions : torch.Tensor
            Ray directions -> dim(B, 3)
        triangle_vertices : torch.Tensor
            Triangle vertices -> dim(N, 3, 3)
        t0 : float
            Minimum t value (default: 0)
        t1 : float
            Maximum t value (default: 100)
            
        Returns
        -------
        beta : torch.Tensor
            beta values -> dim(B, N)
        gamma : torch.Tensor
            gamma values -> dim(B, N)
        t : torch.Tensor
            t values -> dim(B, N)
        intersect : torch.Tensor
            Intersection mask -> dim(B, N)
        
        Notes
        -----
        | x_a - x_b; x_a - x_c; x_d | | beta  |   | x_a - x_e |
        | y_a - y_b; y_a - y_c; y_d | | gamma | = | y_a - y_e |
        | z_a - z_b; z_a - z_c; z_d | | t     |   | z_a - z_e |

        A x = b

        Solve via Cramer's Rule:
            beta  = det(A_1) / det(A)
            gamma = det(A_2) / det(A)
            t     = det(A_3) / det(A)
        Where A_i is A with column i replaced by b
    """
    B = ray_origins.shape[0]
    N = triangle_vertices.shape[0]
    device = triangle_vertices.device
    
    # Setup
    A = torch.zeros((B, N, 3, 3), device=device)
    b = torch.zeros((B, N, 3, 1), device=device)
    triangle_vertices = triangle_vertices[None, :, :, :]
    ray_directions = ray_directions[:, None, :]
    ray_origins = ray_origins[:, None, :]
    
    # 
    A[...,0,0] = triangle_vertices[...,0,0] - triangle_vertices[...,1,0]
    A[...,0,1] = triangle_vertices[...,0,0] - triangle_vertices[...,2,0]
    A[...,0,2] = ray_directions[...,0]
    A[...,1,0] = triangle_vertices[...,0,1] - triangle_vertices[...,1,1]
    A[...,1,1] = triangle_vertices[...,0,1] - triangle_vertices[...,2,1]
    A[...,1,2] = ray_directions[...,1]
    A[...,2,0] = triangle_vertices[...,0,2] - triangle_vertices[...,1,2]
    A[...,2,1] = triangle_vertices[...,0,2] - triangle_vertices[...,2,2]
    A[...,2,2] = ray_directions[...,2]

    b[...,0,0] = triangle_vertices[...,0,0] - ray_origins[...,0]
    b[...,1,0] = triangle_vertices[...,0,1] - ray_origins[...,1]
    b[...,2,0] = triangle_vertices[...,0,2] - ray_origins[...,2]

    # Cramer's Rule
    x     = gffx.linalg.cramer(A, b)
    beta  = x[...,0,0]
    gamma = x[...,1,0]
    t     = x[...,2,0]

    # Intersection Test
    intersect = ((t > t0) & (t < t1)) & ((gamma >= 0) & (gamma <= 1)) & ((beta > 0) & (beta + gamma < 1))

    return beta, gamma, t, intersect

# [4NOW] 1 camera
def mesh_render(
    meshes            : list | gffx.obj.MeshObject,
    camera            : Camera,
    light_intensity   : float               = 1,
    ambient_intensity : float               = 0.2,
    light_pos         : list | torch.Tensor = [5, 5, 5],
    background_color  : list | torch.Tensor = [0, 0, 0],
    ray_chunk_size    : int                 = 4096,
    
    verbose           : bool                = False,
    
    device : Optional[torch.device] = None
):
    # [4NOW] 1 camera  
    B = 1

    if isinstance(meshes, gffx.obj.MeshObject):
        meshes = [meshes]
    
    # [4NOW] Flat colors
    if isinstance(background_color, list):
        background_color = torch.tensor(background_color, device=device, dtype=torch.float32)

    diffuse_colors = [obj.diffuse_color for obj in meshes] + [background_color]
    diffuse_colors = torch.stack(diffuse_colors, dim=0).to(device)

    specular_coefficients = [obj.specular_coefficient for obj in meshes] + [1]
    specular_coefficients = torch.tensor(specular_coefficients, device=device, dtype=torch.float32)

    specular_colors = [obj.specular_color for obj in meshes] + [background_color]
    specular_colors = torch.stack(specular_colors, dim=0).to(device)

    ambient_colors = [obj.ambient_color for obj in meshes] + [background_color]
    ambient_colors = torch.stack(ambient_colors, dim=0).to(device)

    # Light setup
    if isinstance(light_pos, list):
        light_pos  = torch.tensor(light_pos, device=device, dtype=torch.float32)

    # 
    object_hit = torch.full((B * camera.width * camera.height,), -1, device=device, dtype=torch.int64)
    face_hit   = torch.full((B * camera.width * camera.height,), -1, device=device, dtype=torch.int64)
    t_val      = torch.full((B * camera.width * camera.height,), float('inf'), device=device)
    normals    = torch.zeros((B * camera.width * camera.height, 3), device=device)
    hit_pos    = torch.zeros((B * camera.width * camera.height, 3), device=device)

    # 
    if verbose:
        print('Ray Chunk Size:', ray_chunk_size)
    ray_origins    = camera.ray_origins.view(B * camera.width * camera.height, 3)       # dim(B * W * H, 3)
    ray_directions = camera.ray_directions.view(B * camera.width * camera.height, 3) # dim(B * W * H, 3)

    # 
    for obj_idx, obj in enumerate(meshes):
        transformed_vertices, transformed_normals = obj.get_transformed()
        
        face_tri_vertices = transformed_vertices[obj.faces] # dim(F, 3, 3)
        face_tri_normals  = transformed_normals[obj.faces]  # dim(F, 3, 3)
        
        # Chunking
        pbar = tqdm(
            iterable = range(0, B * camera.width * camera.height, ray_chunk_size),
            desc     = f'Object {obj_idx}',
        )
        for i in pbar:
            C = ray_chunk_size if i + ray_chunk_size < B * camera.width * camera.height else (B * camera.width * camera.height - i)
            
            # 
            beta, gamma, t, intersect = gffx.ray.ray_triangle_intersection(
                ray_origins       = ray_origins[i:i+C],    # dim(C, 3)
                ray_directions    = ray_directions[i:i+C], # dim(C, 3)
                triangle_vertices = face_tri_vertices,     # dim(F, 3, 3)
                t0                = 0,
                t1                = 100
            )
            
            # Choose the smallest t which intersects
            t_valid = torch.where(
                intersect,
                torch.where(t < 0, float('inf'), t),
                float('inf')
            )
            face_idx_valid_min = t_valid.argmin(dim=-1, keepdim=True)
            beta_valid_min = torch.gather(
                input = beta,
                dim   = -1,
                index = face_idx_valid_min
            )[:,0]
            gamma_valid_min = torch.gather(
                input = gamma,
                dim   = -1,
                index = face_idx_valid_min
            )[:,0]
            
            t_valid_min = torch.gather(
                input = t_valid,
                dim   = -1,
                index = face_idx_valid_min
            )[:,0]
            intersect_valid_min = torch.gather(
                input = intersect,
                dim   = -1,
                index = face_idx_valid_min
            )[:,0]
            
            # 
            object_hit[i:i+C] = torch.where(
                ((t_valid_min < t_val[i:i+C]) & intersect_valid_min),
                obj_idx,
                object_hit[i:i+C]
            )
            
            #
            face_hit[i:i+C] = torch.where(
                ((t_valid_min < t_val[i:i+C]) & intersect_valid_min),
                face_idx_valid_min[:,0],
                face_hit[i:i+C]
            )
            
            # Interpolate normals
            interpolated_normals = (1 - beta - gamma)[...,None] * face_tri_normals[None, :, 0] + beta[...,None] * face_tri_normals[None, :, 1] + gamma[...,None] * face_tri_normals[None, :, 2]

            interpolated_normals = torch.gather(
                input = interpolated_normals,   # dim(C, F, 3)
                dim   = 1,
                index = face_idx_valid_min[..., None].expand(-1, -1, 3)
            )[:,0,:]
            normals[i:i+C] = torch.where(
                ((t_valid_min < t_val[i:i+C]) & intersect_valid_min)[:,None],
                interpolated_normals,
                normals[i:i+C]
            )
            
            # Interpolate hit positions
            hit_pos[i:i+C] = torch.where(
                ((t_valid_min < t_val[i:i+C]) & intersect_valid_min)[:,None],
                ray_origins[i:i+C] + t_valid_min[:,None] * ray_directions[i:i+C],
                hit_pos[i:i+C]
            )
            
            #
            t_val[i:i+C] = torch.where(
                ((t_valid_min < t_val[i:i+C]) & intersect_valid_min),
                t_valid_min,
                t_val[i:i+C]
            )
        
    # Reshape
    object_hit = object_hit.view(B, camera.width, camera.height)
    face_hit   = face_hit.view(B, camera.width, camera.height)
    normals    = normals.view(B, camera.width, camera.height, 3)
    hit_pos    = hit_pos.view(B, camera.width, camera.height, 3)

    # Compute Phong Shading
    light_pos = light_pos[None, None, None, :]
    light_dir = light_pos - hit_pos
    light_dir /= torch.linalg.norm(light_dir, dim=-1, keepdim=True)

    view_dir  = camera.pos[None, None, None, :] - hit_pos
    view_dir /= torch.linalg.norm(view_dir, dim=-1, keepdim=True)
    bisector_vec  = light_dir + view_dir
    bisector_vec /= torch.linalg.norm(bisector_vec, dim=-1, keepdim=True)

    diffuse_weight  = torch.clamp(torch.sum(light_dir * normals, dim=-1, keepdim=True), min=0)
    specular_weight = torch.clamp(torch.sum(bisector_vec * normals, dim=-1, keepdim=True), min=0) ** specular_coefficients[object_hit][...,None]

    L  = diffuse_colors[object_hit]  * light_intensity * diffuse_weight
    L += specular_colors[object_hit] * light_intensity * specular_weight
    L += ambient_colors[object_hit]  * ambient_intensity

    return L