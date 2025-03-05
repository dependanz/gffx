import torch

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