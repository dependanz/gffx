"""Render the shared Studio B bundle: python studio_b.py studio-b.gffx"""
import sys
import gffx as g

scene = g.Scene.load(sys.argv[1])
world = g.Environment.load(sys.argv[1], samples=128)
bounce = g.IrradianceVolume.bake(
    scene, bounds=((-1.53, .21, .03), (-.69, 1.03, .72)),
    resolution=(3, 3, 3), environment=world, shadow_map_size=2048,
    exclude=("Mover body", "Mover cap", "Mover stripe 0", "Mover stripe 1", "Mover stripe 2"))
probe = g.ReflectionProbe(
    (0, -.1, .38), bounds=((-2, -2.7, -1.62), (2, 2.5, 2.38)),
    exclude=("Chrome sphere", "Glossy sphere", "Planar mirror"),
    resolution=512, samples=128)
mirror = g.PlanarReflection("Planar mirror", (.55, 1.195, .98), (0, -1, 0))
image = g.render(scene, "camera", integrator="raster", supersample=4,
                 shadows=True, shadow_map_size=2048, environment=world,
                 probes=(probe,), irradiance=bounce, mirrors=(mirror,))
g.io.write_png("studio_b.png", image["colour"])
