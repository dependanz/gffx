"""Render an exported studio bundle: python studio_a.py path/to/studio-a.gffx"""
import sys
import gffx as g

scene = g.Scene.load(sys.argv[1])
image = g.render(scene, "camera", integrator="raster", supersample=4,
                 shadows=True, shadow_map_size=2048)
g.io.write_png("studio_a.png", image["colour"])
