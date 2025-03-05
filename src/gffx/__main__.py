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

camera_u  = torch.cross(camera_dir, world_up)
camera_u /= torch.linalg.norm(camera_u)
camera_v  = torch.cross(camera_u, camera_dir)
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
sphere_pos = torch.tensor([-1, -1, 2], device=device, dtype=torch.float32)
sphere_radius = 1

# Check discriminant (ray-sphere intersection)
e_min_c = ray_origins - sphere_pos[None, None, None, :]                    # dim(B, H, W, 3)
d_dot_e_min_c = torch.sum(ray_directions * e_min_c, dim=-1, keepdim=True)  # dim(B, H, W, 1)
d_dot_d = torch.sum(ray_directions * ray_directions, dim=-1, keepdim=True) # dim(B, H, W, 1)
discriminant = d_dot_e_min_c ** 2 - d_dot_d * (torch.sum(e_min_c * e_min_c, dim=-1, keepdim=True) - sphere_radius ** 2)

#
hit_mask = (discriminant >= 0).float()
t = torch.sqrt(discriminant * hit_mask) / d_dot_d
t_minus = torch.clip(-d_dot_e_min_c - t, 0)
t_plus  = torch.clip(-d_dot_e_min_c + t, 0)
t = torch.maximum(t_minus, t_plus) * hit_mask


plt.imshow((t[0,:,:,0]).detach().cpu())
plt.show()


# Compute surface normals

# Set pixel color to value computed from hit point, light position, and normal