import torch

def cramer(
    A : torch.Tensor,
    b : torch.Tensor
) -> torch.Tensor:
    """
        Solve Ax = b via Cramer's Rule
        
        Parameters
        ----------
        A : torch.Tensor
            Coefficient matrices    -> dim(..., N, N)
        b : torch.Tensor
            Right-hand side vectors -> dim(..., N, 1)
    """
    # Get shape for reshape later
    A_shape = A.shape
    
    # Flatten A and b
    A = A.flatten(start_dim=0, end_dim=-3)
    b = b.flatten(start_dim=0, end_dim=-3)
    device = A.device
    
    # Asserts
    B, N, _ = A.shape
    assert N == A.shape[2], "Matrices A must be square"
    assert B == b.shape[0], "Matrices A and vectors b must have the same batch size"
    assert N == b.shape[1], "Matrices A and vectors b must have the same number of rows"
    assert b.shape[2] == 1, "Vectors b must be a column vector"
    
    # Compute determinant of A
    M = torch.linalg.det(A) # dim(B, 1)
    
    # Compute determinants of A_i
    A_dets = torch.zeros((B, N), device=device)
    for i in range(N):
        A_i = A.clone()
        A_i[:, :, i] = b.squeeze(-1)
        A_dets[:, i] = torch.linalg.det(A_i)
    
    # Compute solution
    x = A_dets / M.unsqueeze(-1)
    
    # Reshape x
    x = torch.reshape(x, [*A_shape[:-2], N, 1]) # dim(..., N, 1)
    
    return x

def euler_angles_to_matrix(
    rot_vec : torch.Tensor
):
    """
        Convert Euler angles to rotation matrix
        
        Parameters
        ----------
        rot_vec : torch.Tensor
            Rotation vector -> dim(B, 3)
    """
    B = rot_vec.shape[0]
    device = rot_vec.device
    
    # Rodrigues' rotation formula
    theta = torch.linalg.norm(rot_vec, dim=-1, keepdim=True)[..., None]
    K = torch.zeros((B, 3, 3), device=device)
    K[:,0,1] = -rot_vec[:,2]
    K[:,0,2] =  rot_vec[:,1]
    K[:,1,0] =  rot_vec[:,2]
    K[:,1,2] = -rot_vec[:,0]
    K[:,2,0] = -rot_vec[:,1]
    K[:,2,1] =  rot_vec[:,0]

    R = (
        torch.eye(3, device=device)[None,:].expand(B,-1,-1) 
        + torch.sin(theta) * K 
        + (1 - torch.cos(theta)) * (K @ K)
    )
    R = torch.cat([R, torch.zeros((B, 3, 1), device=device)], dim=-1)
    R = torch.cat([R, torch.zeros((B, 1, 4), device=device)], dim=-2)
    R[:,3,3] = 1
    
    return R

def scale_matrix(
    scale_vec : torch.Tensor
):
    """
        Convert scale vector to scale matrix
        
        Parameters
        ----------
        scale_vec : torch.Tensor
            Scale vector -> dim(B, 3)
    """
    B = scale_vec.shape[0]
    device = scale_vec.device
    
     # Scale matrix
    return torch.cat([
        torch.cat([
            torch.eye(3, device=device)[None,:].expand(B,-1,-1) * scale_vec[..., None], 
            torch.zeros((B, 3, 1), device=device)
        ], dim=-1),
        torch.cat([
            torch.zeros((B, 1, 3), device=device),
            torch.ones((B, 1, 1), device=device)
        ], dim=-1)
    ], dim=-2)
    
def translation_matrix(
    translation_vec : torch.Tensor
):
    """
        Convert translation vector to translation matrix
        
        Parameters
        ----------
        translation_vec : torch.Tensor
            Translation vector -> dim(B, 3)
    """
    B = translation_vec.shape[0]
    device = translation_vec.device
    
     # Translation matrix
    T = torch.eye(4, device=device)[None,:].expand(B,-1,-1).clone()
    T[:,0:3,3] = translation_vec
    
    return T

def transformation_matrix(
    translation_vec : torch.Tensor,
    rotation_vec    : torch.Tensor,
    scale_vec       : torch.Tensor
):
    """
        Convert translation, rotation, and scale vectors to transformation matrix
        
        Parameters
        ----------
        translation_vec : torch.Tensor
            Translation vector -> dim(B, 3)
        rotation_vec    : torch.Tensor
            Rotation vector -> dim(B, 3)
        scale_vec       : torch.Tensor
            Scale vector -> dim(B, 3)
    """
    
    # Rodrigues' rotation formula
    R = euler_angles_to_matrix(rotation_vec)

    # Scale Matrix
    S = scale_matrix(scale_vec)

    # Translation Matrix
    T = translation_matrix(translation_vec)
    
    return T @ R @ S