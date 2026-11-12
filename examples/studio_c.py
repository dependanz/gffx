"""Render an attributed portable Studio C bundle: python studio_c.py scene.gffx"""
import sys
import gffx as g

scene = g.Scene.load(sys.argv[1], device='auto')
image = g.render(scene, 'camera', integrator='raster', supersample=8,
                 shadows=True, shadow_map_size=2048)
g.io.write_png('studio_c.png', image['colour'][0])
