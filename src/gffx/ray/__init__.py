import gffx
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

def ray_triangle_intersection(
    ray_origins, 
    ray_directions, 
    triangle_vertices, 
    t0 = 0, 
    t1 = 100
):
    """
        Compute ray-triangle intersection using Cramer's Rule
        
        Parameters
        ----------
        ray_origins : torch.Tensor
            Ray origins -> dim(B, H, W, 3)
        ray_directions : torch.Tensor
            Ray directions -> dim(B, H, W, 3)
        triangle_vertices : torch.Tensor
            Triangle vertices -> dim(B, 3, 3)
        t0 : float
            Minimum t value (default: 0)
        t1 : float
            Maximum t value (default: 100)
            
        Returns
        -------
        t : torch.Tensor
            t values -> dim(B, H, W, 1)
        intersect : torch.Tensor
            Intersection mask -> dim(B, H, W, 1)
        
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
    screen_width = ray_origins.shape[1]
    screen_height = ray_origins.shape[2]
    device = ray_origins.device

    # Setup
    A = torch.zeros((B, screen_width, screen_height, 3, 3), device=device)
    b = torch.zeros((B, screen_width, screen_height, 3, 1), device=device)

    A[...,0,0] = triangle_vertices[...,0:1,0:1] - triangle_vertices[...,1:2,0:1]
    A[...,0,1] = triangle_vertices[...,0:1,0:1] - triangle_vertices[...,2:3,0:1]
    A[...,0,2] = ray_directions[...,0]
    A[...,1,0] = triangle_vertices[...,0:1,1:2] - triangle_vertices[...,1:2,1:2]
    A[...,1,1] = triangle_vertices[...,0:1,1:2] - triangle_vertices[...,2:3,1:2]
    A[...,1,2] = ray_directions[...,1]
    A[...,2,0] = triangle_vertices[...,0:1,2:3] - triangle_vertices[...,1:2,2:3]
    A[...,2,1] = triangle_vertices[...,0:1,2:3] - triangle_vertices[...,2:3,2:3]
    A[...,2,2] = ray_directions[...,2]

    b[...,0,0] = triangle_vertices[...,0:1,0:1] - ray_origins[...,0]
    b[...,1,0] = triangle_vertices[...,0:1,1:2] - ray_origins[...,1]
    b[...,2,0] = triangle_vertices[...,0:1,2:3] - ray_origins[...,2]

    # Cramer's Rule
    x     = gffx.linalg.cramer(A, b)
    beta  = x[...,0,0]
    gamma = x[...,1,0]
    t     = x[...,2,0]

    # Intersection Test
    intersect = ((t > t0) & (t < t1)) & ((gamma >= 0) & (gamma <= 1)) & ((beta > 0) & (beta + gamma < 1))

    return t, intersect