import torch
from typing import Optional

class Camera:
    def __init__(
        self, 
        width    : int = 512,
        height   : int = 512,
        pos      : list | torch.Tensor = [0, 0, 5], 
        dir      : list | torch.Tensor = [0, 0, -1],
        world_up : list | torch.Tensor = [0, 1, 0],
        ray_origin_scale_du : float = 0,
        ray_origin_scale_dv : float = 0,
        screen_distance : float     = 1.0,
        device   : Optional[torch.device] = None
    ):
        self.width  = width
        self.height = height
        self.ray_origin_scale_du = ray_origin_scale_du
        self.ray_origin_scale_dv = ray_origin_scale_dv
        self.screen_distance = screen_distance
        
        if not device:
            self.device = device
            
        self.pos = pos
        if isinstance(self.pos, list):
            self.pos = torch.tensor(self.pos, device=device, dtype=torch.float32)
        
        self.dir = dir
        if isinstance(self.dir, list):
            self.dir = torch.tensor(self.dir, device=device, dtype=torch.float32)
        
        self.world_up  = world_up
        if isinstance(self.world_up, list):
            self.world_up = torch.tensor(self.world_up, device=device, dtype=torch.float32)
            
        # Camera basis
        self.u  = torch.cross(self.dir, self.world_up, dim=-1)
        self.u /= torch.linalg.norm(self.u)
        self.v  = torch.cross(self.u, self.dir, dim=-1)
        self.w  = -self.dir
        
        # Screen
        self.aspect_ratio = self.width / self.height
        self.left         = -self.aspect_ratio
        self.right        = self.aspect_ratio
        self.bottom       = -1
        self.top          = 1
        
        self.screen_grids = torch.stack(torch.meshgrid(
            (self.right - self.left) * ((torch.arange(0, self.width, device=device) + 0.5) / self.width) + self.left,
            (self.top - self.bottom) * ((torch.arange(0, self.height, device=device) + 0.5) / self.height) + self.bottom,
            indexing='ij'
        ), dim=-1).float().expand(1, -1, -1, -1)

        self.ray_origins  = self.ray_origin_scale_du * self.screen_grids[..., 0:1] * self.u[None, None, None, :]
        self.ray_origins += self.ray_origin_scale_dv * self.screen_grids[..., 1:2] * self.v[None, None, None, :]
        self.ray_origins += self.pos[None,None,None,:]

        # Compute viewing rays
        self.ray_directions  = self.screen_grids[..., 0:1] * self.u[None, None, None, :]
        self.ray_directions += self.screen_grids[..., 1:2] * self.v[None, None, None, :]
        self.ray_directions += self.pos[None,None,None,:] - (self.w[None,None,None,:] * self.screen_distance)
        self.ray_directions -= self.ray_origins
        self.ray_directions /= torch.linalg.norm(self.ray_directions, dim=-1, keepdim=True)