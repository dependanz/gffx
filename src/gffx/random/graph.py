import torch

def adjmat(
    N : int,
    max_degree : int,
    directed : bool = False,
    acyclic  : bool = False
):
    """
        Create random adjacency matrix given number of nodes.
    
        Returns:
        --------
        A : torch.Tensor
            Adjacency matrix of shape (N, N) with binary entries.
    """
    if directed or acyclic:
        raise NotImplementedError("Directed and acyclic graphs are not implemented yet.")
    
    A = torch.zeros((N, N), dtype=torch.float32)
    for i in range(N):
        degree = torch.randint(1, max_degree + 1, (1,)).item()
        neighbors = torch.randperm(N)[:degree]
        A[i, neighbors] = 1.0
    
    return A