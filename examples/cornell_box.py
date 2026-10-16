"""Run with gffx 0.2.1; writes cornell_box.png."""
import gffx as g

scene = g.Scene([
    g.Material.diffuse('white', (.73, .71, .67)),
    g.Material.diffuse('red', (.63, .045, .035)),
    g.Material.diffuse('green', (.065, .43, .12)),
    g.Material.diffuse('ceramic', (.78, .77, .74)),
    g.Mesh.quad('floor', center=(0, 0, .05), u=(2, 0, 0), v=(0, 0, -2.1), material='white'),
    g.Mesh.quad('ceiling', center=(0, 2, .05), u=(2, 0, 0), v=(0, 0, 2.1), material='white'),
    g.Mesh.quad('back', center=(0, 1, -1), u=(2, 0, 0), v=(0, 2, 0), material='white'),
    g.Mesh.quad('left', center=(-1, 1, .05), u=(0, 0, -2.1), v=(0, 2, 0), material='red'),
    g.Mesh.quad('right', center=(1, 1, .05), u=(0, 0, 2.1), v=(0, 2, 0), material='green'),
    g.Mesh.sphere('ball', radius=.5, center=(-.07, .5, -.15), material='ceramic'),
    g.Light.rectangle('ceiling_light', center=(0, 1.98, -.15),
                      u=(.7, 0, 0), v=(0, 0, .6), radiance=(15, 14.4, 13.2)),
]).to('auto')

camera = g.Camera.look_at((0, 1.04, 3.9), (0, .99, -.15),
                          size=(512, 512), fov=40, device='auto')
image = g.render(scene, camera, shadows=True, ambient=.08)
g.io.write_png('cornell_box.png', image['colour'])
