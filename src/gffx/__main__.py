from typing import Optional
import gffx
import torch
import argparse
import math
import matplotlib.pyplot as plt

import gffx.linalg

################################################################
# ARGUMENT PARSER
################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
# [DEBUG] RAY TRACING
################################################################

#
B = 2

# Setup screen
screen_width  = 720
screen_height = 480
screens = torch.zeros((B, screen_width, screen_height, 3), device=device)

# Setup camera
camera_pos          = torch.tensor([0, 0, 3], device=device, dtype=torch.float32)
camera_dir          = torch.tensor([0, 0, -1], device=device, dtype=torch.float32)
camera_dir         /= torch.linalg.norm(camera_dir)

world_up            = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

camera_u  = torch.cross(camera_dir, world_up, dim=-1)
camera_u /= torch.linalg.norm(camera_u)
camera_v  = torch.cross(camera_u, camera_dir, dim=-1)
camera_w  = -camera_dir

# Ray origins
aspect_ratio = screen_width / screen_height
left         = -aspect_ratio
right        = aspect_ratio
bottom       = -1
top          = 1

screen_grids = torch.stack(torch.meshgrid(
    (right - left) * ((torch.arange(0, screen_width, device=device) + 0.5) / screen_width) + left,
    (top - bottom) * ((torch.arange(0, screen_height, device=device) + 0.5) / screen_height) + bottom,
    indexing='ij'
), dim=-1).float().expand(B, -1, -1, -1)

ray_origin_scale_du = 0
ray_origin_scale_dv = 0

ray_origins  = ray_origin_scale_du * screen_grids[..., 0:1] * camera_u[None, None, None, :]
ray_origins += ray_origin_scale_dv * screen_grids[..., 1:2] * camera_v[None, None, None, :]
ray_origins += camera_pos[None,None,None,:]

# Compute viewing rays
distance = 1
ray_directions  = screen_grids[..., 0:1] * camera_u[None, None, None, :]
ray_directions += screen_grids[..., 1:2] * camera_v[None, None, None, :]
ray_directions += camera_pos[None,None,None,:] - (camera_w[None,None,None,:] * distance)
ray_directions -= ray_origins
ray_directions /= torch.linalg.norm(ray_directions, dim=-1, keepdim=True)

################################################################
# [CASE] Ray-Sphere Intersection
################################################################

# Object setup
# sphere_pos = torch.tensor([-1, -1, 2], device=device, dtype=torch.float32)
# sphere_radius = 1

# t = gffx.ray.ray_sphere_intersection(ray_origins, ray_directions, sphere_pos, sphere_radius)
# plt.imshow((t[0,:,:,0]).detach().cpu())
# plt.show()

################################################################
# [CASE] Ray-Triangle Intersection (4.4.2)
################################################################

# Object setup
triangle_vertices = torch.tensor([
    [-0.5, -0.5, 0],
    [ 0.5, -0.5, 0],
    [ 0,  0.5, 0]
], device=device, dtype=torch.float32)[None,...].expand(B,-1,-1) # dim(3, 3)

# Model Transformation (Translation, Orientation, Scale)
translation = torch.tensor([0.5, 0, 0], device=device, dtype=torch.float32)[None,:].expand(B,-1)
rotation    = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)[None,:].expand(B,-1)
scale       = torch.tensor([1, 1, 1], device=device, dtype=torch.float32)[None,:].expand(B,-1)

# Model Transformation Matrix
M = gffx.linalg.transformation_matrix(
    translation_vec = translation,
    rotation_vec    = rotation,
    scale_vec       = scale
)

# Transform triangle vertices
triangle_vertices_h = torch.cat([triangle_vertices, torch.ones((B, 3, 1), device=device)], dim=-1) # dim(B, 3, 4)
triangle_vertices_h = triangle_vertices_h @ M.transpose(-1, -2) # dim(B, 3, 4)
transfomed_triangle_vertices = triangle_vertices_h[..., 0:3]

t, intersect = gffx.ray.ray_triangle_intersection(
    ray_origins = ray_origins,
    ray_directions = ray_directions,
    triangle_vertices = transfomed_triangle_vertices,
    t0 = 0,
    t1 = 100
)
plt.imshow((t[0].cpu() * intersect[0].cpu()).T)
plt.gca().invert_yaxis()
plt.show()


################################################################
# [CASE] Multiple Objects (4.4.4)
################################################################

# Object setup
object_list = []
object_list.append(
    gffx.obj.generate_cube_mesh(
        init_translation = [0, 0, 2],
        init_rotation    = [0, 0, 0],
        init_scale       = [1, 1, 1],
        device           = device
    )
)

breakpoint()