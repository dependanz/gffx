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
screen_width  = 512
screen_height = 512
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
# triangle_vertices = torch.tensor([
#     [-0.5, -0.5, 0],
#     [ 0.5, -0.5, 0],
#     [ 0,    0.5, 0]
# ], device=device, dtype=torch.float32)[None,...].expand(B,-1,-1) # dim(3, 3)

# Model Transformation (Translation, Orientation, Scale)
# translation = torch.tensor([0.5, 0, 0], device=device, dtype=torch.float32)[None,:].expand(B,-1)
# rotation    = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)[None,:].expand(B,-1)
# scale       = torch.tensor([1, 1, 1], device=device, dtype=torch.float32)[None,:].expand(B,-1)

# Model Transformation Matrix
# M = gffx.linalg.transformation_matrix(
#     translation_vec = translation,
#     rotation_vec    = rotation,
#     scale_vec       = scale
# )

# Transform triangle vertices
# triangle_vertices_h = torch.cat([triangle_vertices, torch.ones((B, 3, 1), device=device)], dim=-1) # dim(B, 3, 4)
# triangle_vertices_h = triangle_vertices_h @ M.transpose(-1, -2) # dim(B, 3, 4)
# transfomed_triangle_vertices = triangle_vertices_h[..., 0:3]

# t, intersect = gffx.ray.ray_triangle_intersection(
#     ray_origins = ray_origins,
#     ray_directions = ray_directions,
#     triangle_vertices = transfomed_triangle_vertices,
#     t0 = 0,
#     t1 = 100
# )
# plt.imshow((t[0].cpu() * intersect[0].cpu()).T)
# plt.gca().invert_yaxis()
# plt.show()

# ################################################################
# # [CASE] Multiple Objects (4.4.4)
# ################################################################

# # Object setup
# face_vertices = torch.load('/home/danzieboy/dev/face-animation/PDGA/PDGA/assets/cache/vocaset/motionspace/template_vertices.pt')[0].to(device)
# face_faces    = torch.load('/home/danzieboy/dev/face-animation/PDGA/PDGA/assets/cache/vocaset/motionspace/template_faces.pt')[0].to(device)
# face_vertices_mean = torch.mean(face_vertices, dim=0)
# face_vertices_centered = face_vertices - face_vertices_mean
# face_vertices_scale = torch.max(torch.linalg.norm(face_vertices_centered, dim=-1))
# face_vertices = face_vertices / face_vertices_scale

# object_list = [
#     # gffx.obj.generate_cube_mesh(
#     #     init_translation = [0, 0, 0],
#     #     init_rotation    = [0, 0, 0],
#     #     init_scale       = [1, 1, 1],
#     #     device           = device
#     # ),
#     # gffx.obj.generate_icosphere_mesh(
#     #     num_subdivisions = 2,
#     #     init_translation = [0, 0, 0],
#     #     init_rotation    = [0, 0, 0],
#     #     init_scale       = [1, 1, 1],
#     #     device           = device
#     # )
#     gffx.obj.mesh_from_vertices_and_faces(
#         vertices             = face_vertices,
#         faces                = face_faces,
#             init_translation = [0, 0.0, 2],
#             init_rotation    = [0, 0, 0],
#             init_scale       = [1, 1, 1],
#             device           = device
#     )
# ]

# # [4NOW] Flat colors
# background_color = [0, 0, 0]
# diffuse_color = [
#     # [0.3, 0.3, 1],
#     [0.5, 0.5, 0.5],
# ]
# diffuse_color.append(background_color)
# diffuse_color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)

# specular_coefficient = 1
# specular_color = [
#     # [0.3, 0.3, 1],
#     [0.5, 0.5, 0.5],
# ]
# specular_color.append(background_color)
# specular_color = torch.tensor(specular_color, device=device, dtype=torch.float32)

# ambient_color = [
#     # [0.3, 0.3, 1],
#     [0.5, 0.5, 0.5],
# ]
# ambient_color.append(background_color)
# ambient_color = torch.tensor(ambient_color, device=device, dtype=torch.float32)

# # Light setup
# light_intensity = 1
# ambient_intensity = 0.2
# light_pos = torch.tensor([5, 5, 5], device=device, dtype=torch.float32)

# object_hit = torch.full((B, screen_width, screen_height), -1, device=device, dtype=torch.int64)
# face_hit   = torch.full((B, screen_width, screen_height), -1, device=device, dtype=torch.int64)
# t_val      = torch.full((B, screen_width, screen_height), float('inf'), device=device)
# normals    = torch.zeros((B, screen_width, screen_height, 3), device=device)
# hit_pos    = torch.zeros((B, screen_width, screen_height, 3), device=device)
# for obj_idx, obj in enumerate(object_list):
#     transformed_vertices, transformed_normals = obj.get_transformed()
#     for face_idx, tri in enumerate(obj.faces):
#         triangle_vertices = transformed_vertices[tri][None,...] # dim(1, 3, 3)
#         triangle_normals  = transformed_normals[tri][None, ...] # dim(1, 3, 3)
        
#         # 
#         beta, gamma, t, intersect = gffx.ray.ray_triangle_intersection(
#             ray_origins       = ray_origins,
#             ray_directions    = ray_directions,
#             triangle_vertices = triangle_vertices,
#             t0                = 0,
#             t1                = 100
#         )
#         beta  = beta[...,None]
#         gamma = gamma[...,None]
        
#         # 
#         object_hit = torch.where(
#             (t < t_val) & intersect,
#             obj_idx,
#             object_hit
#         )
        
#         #
#         face_hit = torch.where(
#             (t < t_val) & intersect,
#             face_idx,
#             face_hit
#         )
        
#         # Interpolate normals
#         interpolated_normals = (1 - beta - gamma) * triangle_normals[:,None,None,0] + beta * triangle_normals[:,None,None,1] + gamma * triangle_normals[:,None,None,2]
#         normals = torch.where(
#             ((t < t_val) & intersect)[...,None],
#             interpolated_normals,
#             normals
#         )
        
#         # Interpolate hit positions
#         hit_pos = torch.where(
#             ((t < t_val) & intersect)[...,None],
#             ray_origins + t[...,None] * ray_directions,
#             hit_pos
#         )
        
#         #
#         t_val = torch.where(
#             (t < t_val) & intersect,
#             t,
#             t_val
#         )
        
#         print(obj_idx, face_idx)

# # Compute Lambertian shading
# # light_pos = light_pos[None, None, None, :]
# # light_dir = light_pos - hit_pos
# # light_dir /= torch.linalg.norm(light_dir, dim=-1, keepdim=True)

# # light_dir_dot_normals = torch.sum(light_dir * normals, dim=-1, keepdim=True) # dim(B, H, W, 1)
# # light_dir_dot_normals = light_intensity * torch.clamp(light_dir_dot_normals, min=0)

# # L = diffuse_color[object_hit] * light_dir_dot_normals

# # Compute Phong Shading
# light_pos = light_pos[None, None, None, :]
# light_dir = light_pos - hit_pos
# light_dir /= torch.linalg.norm(light_dir, dim=-1, keepdim=True)

# view_dir = camera_pos[None, None, None, :] - hit_pos
# view_dir /= torch.linalg.norm(view_dir, dim=-1, keepdim=True)
# bisector_vec = light_dir + view_dir
# bisector_vec /= torch.linalg.norm(bisector_vec, dim=-1, keepdim=True)

# diffuse_weight = torch.clamp(torch.sum(light_dir * normals, dim=-1, keepdim=True), min=0)
# specular_weight = torch.clamp(torch.sum(bisector_vec * normals, dim=-1, keepdim=True), min=0) ** specular_coefficient

# L = diffuse_color[object_hit] * light_intensity * diffuse_weight
# L += specular_color[object_hit] * light_intensity * specular_weight
# L += ambient_color[object_hit] * ambient_intensity

# plt.imshow((L[0].cpu()).permute(1, 0, 2))
# plt.gca().invert_yaxis()
# plt.show()

################################################################
# FASTER ALGORITHM
################################################################

# Object setup
face_vertices = torch.load('/home/danzieboy/dev/face-animation/PDGA/PDGA/assets/cache/vocaset/motionspace/template_vertices.pt')[0].to(device)
face_faces    = torch.load('/home/danzieboy/dev/face-animation/PDGA/PDGA/assets/cache/vocaset/motionspace/template_faces.pt')[0].to(device)
face_vertices_mean = torch.mean(face_vertices, dim=0)
face_vertices_centered = face_vertices - face_vertices_mean
face_vertices_scale = torch.max(torch.linalg.norm(face_vertices_centered, dim=-1))
face_vertices = face_vertices / face_vertices_scale

object_list = [
    # gffx.obj.generate_cube_mesh(
    #     init_translation = [0, 0, 0],
    #     init_rotation    = [0, 0, 0],
    #     init_scale       = [1, 1, 1],
    #     device           = device
    # ),
    # gffx.obj.generate_icosphere_mesh(
    #     num_subdivisions = 2,
    #     init_translation = [0, 0, 0],
    #     init_rotation    = [0, 0, 0],
    #     init_scale       = [1, 1, 1],
    #     device           = device
    # )
    gffx.obj.mesh_from_vertices_and_faces(
        vertices             = face_vertices,
        faces                = face_faces,
            init_translation = [0, 0.0, 2],
            init_rotation    = [0, 0, 0],
            init_scale       = [1, 1, 1],
            device           = device
    )
]

# [4NOW] Flat colors
background_color = [0, 0, 0]
diffuse_color = [
    # [0.3, 0.3, 1],
    [0.5, 0.5, 0.5],
]
diffuse_color.append(background_color)
diffuse_color = torch.tensor(diffuse_color, device=device, dtype=torch.float32)

specular_coefficient = 1
specular_color = [
    # [0.3, 0.3, 1],
    [0.5, 0.5, 0.5],
]
specular_color.append(background_color)
specular_color = torch.tensor(specular_color, device=device, dtype=torch.float32)

ambient_color = [
    # [0.3, 0.3, 1],
    [0.5, 0.5, 0.5],
]
ambient_color.append(background_color)
ambient_color = torch.tensor(ambient_color, device=device, dtype=torch.float32)

# Light setup
light_intensity = 1
ambient_intensity = 0.2
light_pos  = torch.tensor([5, 5, 5], device=device, dtype=torch.float32)

object_hit = torch.full((B,  screen_width, screen_height), -1, device=device, dtype=torch.int64)
face_hit   = torch.full((B,  screen_width, screen_height), -1, device=device, dtype=torch.int64)
t_val      = torch.full((B,  screen_width, screen_height), float('inf'), device=device)
normals    = torch.zeros((B, screen_width, screen_height, 3), device=device)
hit_pos    = torch.zeros((B, screen_width, screen_height, 3), device=device)
for obj_idx, obj in enumerate(object_list):
    transformed_vertices, transformed_normals = obj.get_transformed()
    
    face_tri_vertices = transformed_vertices[obj.faces] # dim(F, 3, 3)
    face_tri_normals  = transformed_normals[obj.faces]  # dim(F, 3, 3)
    
    B, H, W, _ = ray_origins.shape
    breakpoint()
    beta, gamma, t, intersect = gffx.ray.ray_triangle_intersection(
        ray_origins       = ray_origins.view(B * H * W, 3),    # dim(B * H * W, 3)
        ray_directions    = ray_directions.view(B * H * W, 3), # dim(B * H * W, 3)
        triangle_vertices = face_tri_vertices,                 # dim(F, 3, 3)
        t0                = 0,
        t1                = 100
    ) # dim(B * H * W, F) -> dim(B, H, W, F)
    
    breakpoint()
    
    for face_idx, tri in enumerate(obj.faces):
        triangle_vertices = transformed_vertices[tri][None, ...] # dim(1, 3, 3)
        triangle_normals  = transformed_normals[tri][None, ...] # dim(1, 3, 3)
        
        # 
        beta, gamma, t, intersect = gffx.ray.ray_triangle_intersection(
            ray_origins       = ray_origins,
            ray_directions    = ray_directions,
            triangle_vertices = triangle_vertices,
            t0                = 0,
            t1                = 100
        )
        beta  = beta[...,None]
        gamma = gamma[...,None]
        
        # 
        object_hit = torch.where(
            (t < t_val) & intersect,
            obj_idx,
            object_hit
        )
        
        #
        face_hit = torch.where(
            (t < t_val) & intersect,
            face_idx,
            face_hit
        )
        
        # Interpolate normals
        interpolated_normals = (1 - beta - gamma) * triangle_normals[:,None,None,0] + beta * triangle_normals[:,None,None,1] + gamma * triangle_normals[:,None,None,2]
        normals = torch.where(
            ((t < t_val) & intersect)[...,None],
            interpolated_normals,
            normals
        )
        
        # Interpolate hit positions
        hit_pos = torch.where(
            ((t < t_val) & intersect)[...,None],
            ray_origins + t[...,None] * ray_directions,
            hit_pos
        )
        
        #
        t_val = torch.where(
            (t < t_val) & intersect,
            t,
            t_val
        )
        
        print(obj_idx, face_idx)

# Compute Lambertian shading
# light_pos = light_pos[None, None, None, :]
# light_dir = light_pos - hit_pos
# light_dir /= torch.linalg.norm(light_dir, dim=-1, keepdim=True)

# light_dir_dot_normals = torch.sum(light_dir * normals, dim=-1, keepdim=True) # dim(B, H, W, 1)
# light_dir_dot_normals = light_intensity * torch.clamp(light_dir_dot_normals, min=0)

# L = diffuse_color[object_hit] * light_dir_dot_normals

# Compute Phong Shading
light_pos = light_pos[None, None, None, :]
light_dir = light_pos - hit_pos
light_dir /= torch.linalg.norm(light_dir, dim=-1, keepdim=True)

view_dir = camera_pos[None, None, None, :] - hit_pos
view_dir /= torch.linalg.norm(view_dir, dim=-1, keepdim=True)
bisector_vec = light_dir + view_dir
bisector_vec /= torch.linalg.norm(bisector_vec, dim=-1, keepdim=True)

diffuse_weight = torch.clamp(torch.sum(light_dir * normals, dim=-1, keepdim=True), min=0)
specular_weight = torch.clamp(torch.sum(bisector_vec * normals, dim=-1, keepdim=True), min=0) ** specular_coefficient

L = diffuse_color[object_hit] * light_intensity * diffuse_weight
L += specular_color[object_hit] * light_intensity * specular_weight
L += ambient_color[object_hit] * ambient_intensity

plt.imshow((L[0].cpu()).permute(1, 0, 2))
plt.gca().invert_yaxis()
plt.show()