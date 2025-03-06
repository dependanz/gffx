### gffx - a minimal library for (differentiable) graphics

```Python
import gffx
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Camera
camera = gffx.ray.Camera(
    width   = 512,
    height  = 512,
    pos = [0, 0, 3],
    dir = [0, 0, -1],

    device = device
)

# Setup list of meshes
vertices          = torch.load('path/to/vertices.pt').to(device)
faces             = torch.load('path/to/vertices.pt').to(device)

object_list = [
    gffx.obj.mesh_from_vertices_and_faces(
        vertices             = vertices,
        faces                = faces,
        init_translation     = [0, 0.0, 2],
        init_rotation        = [0, 0, 0],
        init_scale           = [1, 1, 1],
        
        ambient_color        = [0.5, 0.5, 0.5],
        diffuse_color        = [0.5, 0.5, 0.5],
        specular_color       = [0.5, 0.5, 0.5],
        specular_coefficient = 1,
        
        device           = device
    )
]

# Ray Trace Render
images = gffx.ray.mesh_render(
    meshes = object_list,
    camera = camera,
    light_intensity = 1.0,
    ambient_intensity = 0.2,
    light_pos = [5, 5, 5],
    background_color = [0, 0, 0],
    ray_chunk_size = 4096,
    
    device = device
)

plt.imshow((images[0].cpu()).permute(1, 0, 2))
plt.gca().invert_yaxis()
plt.show()
```

### [WIP] CLI
```
python -m gffx --vertices '/path/to/vertices.pt' --faces 'path/to/faces.pt'
```

### Motivation
1. Installing PyTorch3D everytime in colab is a hassle. This library's first aim is to render a mesh using native operations in PyTorch.
