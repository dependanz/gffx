"""Bounded connection-local mesh reuse; only negotiated pages receive deltas."""
from collections import OrderedDict


class MeshDeltaCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.paths = OrderedDict()

    def clear(self):
        self.paths.clear(); self.bytes = 0

    def encode(self, header, arrays):
        if header.get("kind") != "mesh":
            return header, arrays
        path = header["path"]
        old = self.paths.pop(path, None)
        if old:
            self.bytes -= old[2]
        previous = old[1] if old else {}
        sent = []
        for name, view in arrays:
            prior = previous.get(name)
            if name == "vertices" or prior is None or (view.dtype, view.shape, view.data) != (prior.dtype, prior.shape, prior.data):
                sent.append((name, view))
        out = dict(header)
        if old:
            out.update(mesh_delta=True, base_step=old[0], mesh_fields=[name for name, _ in arrays])
        saved = {name: view for name, view in arrays if name != "vertices"}
        size = sum(len(view.data) for view in saved.values())
        while self.paths and self.bytes + size > self.max_bytes:
            _, removed = self.paths.popitem(last=False); self.bytes -= removed[2]
        if size <= self.max_bytes:
            self.paths[path] = (header["step"], saved, size); self.bytes += size
        return out, sent
