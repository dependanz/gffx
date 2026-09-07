"""Finite synthetic visual demonstration. No training loop or background mesh generator."""
import struct
from urllib.parse import urlsplit
from gffx.dashboard import ArrayView, connect
from .reporter import Reporter

def publish(address, run):
    reporter = Reporter(address,run)
    if not reporter.send("start",kind="demo"): raise RuntimeError(reporter.error)
    u = urlsplit(address); port = u.port or (443 if u.scheme=="https" else 80)
    vertices = ArrayView("float32", (3,3), struct.pack("<9f",-.6,-.5,0,.6,-.5,0,0,.6,0))
    faces = ArrayView("uint32", (1,3), struct.pack("<3I",0,1,2))
    colors = ArrayView("float32", (3,3), struct.pack("<9f",1,80/255,40/255,40/255,220/255,80/255,60/255,100/255,1))
    image = ArrayView("uint8", (2,2,3), bytes([255,80,40,40,220,80,60,100,255,240,230,90]))
    with connect("%s://%s:%d"%(u.scheme,u.hostname,port),run=run,name="finite synthetic demo",wait_acks=True) as dash:
        # Exactly one complete checkpoint, retained in a configurable recent queue.
        dash.queue("demo/triangle","mesh",recent_window=100,snapshot_every=1000).push(vertices,faces,colors=colors,step=0)
        dash.image("demo/image",image,step=0)
        for step in range(20): dash.scalar("demo/curve",1/(step+1),step=step)
        dash.text("demo/description","Panel playground: tap a triangle vertex to highlight its matching RGB image swatch (red top-left, green top-right, blue bottom-left). Hover or tap the image to inspect a pixel. The curve is synthetic 1/(step+1), not measured training. Minimal producer: tools/testing_dashboard/demo.py uses dash.queue(...).push(vertices, faces, colors=colors, step=0), dash.image(...), and dash.scalar(...). These are demonstrations, not operation test results.",step=0)
    artifacts=[{"kind":kind,"path":"demo/"+name} for kind,name in
               [("mesh","triangle"),("image","image"),("plot","curve"),("text","description")]]
    if not reporter.send("demo",title="Shared panel demonstration",artifacts=artifacts): raise RuntimeError(reporter.error)
    if not reporter.send("finish",exit_code=0): raise RuntimeError(reporter.error)
    print("Demo:",run)
