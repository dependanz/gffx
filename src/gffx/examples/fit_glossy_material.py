"""Recover RGB albedo and roughness from a synthetic image with known lighting."""
import torch
import gffx as g

albedo = torch.tensor([.55, .25, .15], requires_grad=True)
roughness = torch.tensor(.65, requires_grad=True)
camera = g.Camera.look_at((0, 0, 3), (0, 0, 0), size=(64, 64), device='auto')

def render(albedo, roughness):
    scene = g.Scene([
        g.Material.glossy('surface', albedo, roughness=roughness, specular=.35),
        g.Mesh.sphere('ball', radius=.7, material='surface'),
        g.Light.rectangle('light', center=(1, 1.2, 2), u=(.8, 0, 0),
                          v=(0, -.8, 0), radiance=(12, 12, 12), samples=2, visible=False),
    ]).to('auto')
    return g.render(scene, camera)['colour']

target = render(torch.tensor([.22, .45, .70]), torch.tensor(.32)).detach()
g.io.write_png('gradient_target.png', target)
g.io.write_png('gradient_initial.png', render(albedo, roughness))
optimizer = torch.optim.Adam([albedo, roughness], lr=.03)
for step in range(100):
    optimizer.zero_grad()
    loss = (render(albedo, roughness) - target).square().mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        albedo.clamp_(0, 1)
        roughness.clamp_(.05, 1)
g.io.write_png('gradient_fitted.png', render(albedo, roughness))
print(f'Albedo {albedo.detach().tolist()}, roughness {roughness.item():.4f}, loss {loss.item():.3g}')
