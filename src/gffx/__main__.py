import gffx
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

################################################################
# ARGUMENT PARSER
################################################################
parser = argparse.ArgumentParser(description='gffx')
parser.add_argument(
    '--vertices',
    type=Path,
    help='Path to vertices PyTorch file (*.pt)',
    required=True
)
parser.add_argument(
    '--faces',
    type=Path,
    help='Path to faces PyTorch file (*.pt)',
    required=True
)
parser.add_argument(
    '--center_and_scale',
    action  = 'store_true',
    help    = 'Center and scale object',
    default = False
)

parser.add_argument(
    '--camera_width',
    type=int,
    help='Camera width',
    default=512
)
parser.add_argument(
    '--camera_height',
    type=int,
    help='Camera height',
    default=512
)
parser.add_argument(
    '--camera_pos',
    type=list,
    help='Camera position',
    default=[0, 0, 1.2]
)
parser.add_argument(
    '--camera_dir',
    type=list,
    help='Camera direction',
    default=[0, 0, -1]
)
parser.add_argument(
    '--camera_screen_distance',
    type    = float,
    help    = 'Camera screen distance',
    default = 1.0
)

parser.add_argument(
    '--ambient_color',
    type=list,
    help='Ambient color',
    default=[0.5, 0.5, 0.5]
)
parser.add_argument(
    '--diffuse_color',
    type=list,
    help='Diffuse color',
    default=[0.5, 0.5, 0.5]
)
parser.add_argument(
    '--specular_color',
    type=list,
    help='Specular color',
    default=[0.5, 0.5, 0.5]
)
parser.add_argument(
    '--specular_coefficient',
    type=float,
    help='Specular coefficient',
    default=1
)
parser.add_argument(
    '--light_intensity',
    type=float,
    help='Light intensity',
    default=1
)
parser.add_argument(
    '--ambient_intensity',
    type=float,
    help='Ambient intensity',
    default=0.2
)
parser.add_argument(
    '--light_pos',
    type=list,
    help='Light position',
    default=[5, 5, 5]
)
parser.add_argument(
    '--background_color',
    type=list,
    help='Background color',
    default=[0, 0, 0]
)
parser.add_argument(
    '--ray_chunk_size',
    type=int,
    help='Ray chunk size',
    default=1024
)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Camera
camera = gffx.ray.Camera(
    width  = args.camera_width,
    height = args.camera_height,
    pos    = args.camera_pos,
    dir    = args.camera_dir,
    screen_distance = args.camera_screen_distance,

    device = device
)

# Setup list of meshes
vertices          = torch.load(args.vertices)[0].to(device)
faces             = torch.load(args.faces)[0].to(device)

if args.center_and_scale:
    vertices_mean     = torch.mean(vertices, dim=0)
    vertices_centered = vertices - vertices_mean
    vertices_scale    = torch.max(torch.linalg.norm(vertices_centered, dim=-1))
    vertices          = vertices / vertices_scale

object_list = [
    gffx.obj.mesh_from_vertices_and_faces(
        vertices             = vertices,
        faces                = faces,
        init_translation     = [0, 0.0, 0],
        init_rotation        = [0, 0, 0],
        init_scale           = [1, 1, 1],
        
        ambient_color        = args.ambient_color,
        diffuse_color        = args.diffuse_color,
        specular_color       = args.specular_color,
        specular_coefficient = args.specular_coefficient,
        
        device           = device
    )
]

# Ray Trace Render
images = gffx.ray.mesh_render(
    meshes = object_list,
    camera = camera,
    light_intensity     = args.light_intensity,
    ambient_intensity   = args.ambient_intensity,
    light_pos           = args.light_pos,
    background_color    = args.background_color,
    ray_chunk_size      = args.ray_chunk_size,
    
    device = device
)

plt.imshow((images[0].cpu()).permute(1, 0, 2))
plt.gca().invert_yaxis()
plt.show()