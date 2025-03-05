import gffx
import torch
import argparse
import matplotlib.pyplot as plt

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
screen_width  = 512
screen_height = 512
screens = torch.zeros((B, screen_height, screen_width, 3), device=device)

# Setup camera
camera_pos          = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
camera_dir          = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
camera_dir         /= torch.linalg.norm(camera_dir)

world_up            = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

camera_u  = torch.cross(camera_dir, world_up, dim=-1)
camera_u /= torch.linalg.norm(camera_u)
camera_v  = torch.cross(camera_u, camera_dir, dim=-1)
camera_w  = -camera_dir

# Ray origins
left   = -1
right  = 1
bottom = -1
top    = 1

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
    [-1, -1, 2],
    [ 1, -1, 2],
    [ 0,  1, 2]
], device=device, dtype=torch.float32)[None,...].expand(B,-1,-1) # dim(3, 3)

# Model Transformation (Translation, Orientation, Scale)
translation = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)[None,:].expand(B,-1)
rotation    = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)[None,:].expand(B,-1)
scale       = torch.tensor([1, 1, 1], device=device, dtype=torch.float32)[None,:].expand(B,-1)

# Rodrigues' rotation formula
theta = torch.linalg.norm(rotation, dim=-1, keepdim=True)[..., None]
K = torch.zeros((B, 3, 3), device=device)
K[:,0,1] = -rotation[:,2]
K[:,0,2] =  rotation[:,1]
K[:,1,0] =  rotation[:,2]
K[:,1,2] = -rotation[:,0]
K[:,2,0] = -rotation[:,1]
K[:,2,1] =  rotation[:,0]

R = (
    torch.eye(3, device=device)[None,:].expand(B,-1,-1) 
    + torch.sin(theta) * K 
    + (1 - torch.cos(theta)) * (K @ K)
)
R = torch.cat([R, torch.zeros((B, 3, 1), device=device)], dim=-1)
R = torch.cat([R, torch.zeros((B, 1, 4), device=device)], dim=-2)
R[:,3,3] = 1

# Scale Matrix
S = torch.cat([
     torch.cat([
    torch.eye(3, device=device)[None,:].expand(B,-1,-1) * scale[..., None], 
    torch.zeros((B, 3, 1), device=device)], dim=-1), torch.zeros((B, 1, 4), device=device)], dim=-2)

# Translation Matrix
T = torch.eye(4, device=device)[None,:].expand(B,-1,-1)
T[:,0:3,3] = translation

# Model Transformation Matrix
M = T @ R @ S

breakpoint()