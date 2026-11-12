"""Experimental guarded short-flight recovery for glass edge neighbors."""
import torch


def _glass_edge_neighbors(faces, face_glass):
    triangles = faces.detach().cpu().tolist()
    glass = face_glass.detach().cpu().tolist()
    adjacency = [[-1, -1, -1] for _ in triangles]
    pending = {}

    for fid, (tri, active) in enumerate(zip(triangles, glass)):
        if not active:
            continue
        edges = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        for eid, (a, b) in enumerate(edges):
            key = (min(a, b), max(a, b))
            if key in pending:
                other, other_eid = pending.pop(key)
                adjacency[fid][eid] = other
                adjacency[other][other_eid] = fid
            else:
                pending[key] = (fid, eid)

    return torch.tensor(adjacency, dtype=torch.long, device=faces.device)


class EdgeShortFlight:
    def __init__(self, vertices, faces, owners, glass_faces, normals):
        self.vertices64 = vertices.to(torch.float64)
        self.faces = faces
        self.owners = owners
        self.normals = normals
        self.adjacent = _glass_edge_neighbors(faces, glass_faces)

    def recover(self, hit, origin, direction, minimum, maximum,
                previous, medium):
        # The qualified double-precision path is unchanged.
        if origin.dtype != torch.float32:
            return hit

        valid_previous = previous >= 0
        if not bool(valid_previous.any()):
            return hit

        rows = valid_previous.nonzero(as_tuple=True)[0]
        adjacent = self.adjacent[previous[rows]]
        possible = adjacent >= 0

        if not bool(possible.any()):
            return hit

        ids = adjacent.clamp_min(0)
        tri = self.vertices64[self.faces[ids].long()]
        a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
        ab, ac = b - a, c - a

        o = origin[rows].double()[:, None, :]
        d = direction[rows].double()[:, None, :]

        normal = torch.linalg.cross(ab, ac)
        p = torch.linalg.cross(d.expand_as(ac), ac)
        determinant = (ab * p).sum(-1)

        eps = torch.finfo(torch.float64).eps
        nonparallel = determinant.abs() > (
            16 * eps * torch.linalg.vector_norm(normal, dim=-1)
        )
        inverse = 1.0 / torch.where(nonparallel, determinant, 1.0)

        delta = o - a
        u = (delta * p).sum(-1) * inverse
        q = torch.linalg.cross(delta, ab)
        v = (d * q).sum(-1) * inverse
        t = (ac * q).sum(-1) * inverse

        lower = torch.as_tensor(
            minimum, dtype=origin.dtype, device=origin.device
        ).expand(len(origin))[rows, None].double()

        upper = torch.as_tensor(
            maximum, dtype=origin.dtype, device=origin.device
        ).expand(len(origin))[rows, None].double()

        entering = (d * normal).sum(-1) < 0
        owner = self.owners[ids]
        current = medium[rows, None]

        # Retain strict single-medium semantics.
        medium_ok = torch.where(
            current < 0,
            entering,
            (~entering) & (owner == current),
        )

        eligible = (
            possible
            & nonparallel
            & medium_ok
            & torch.isfinite(t)
            & (u >= -16 * eps)
            & (v >= -16 * eps)
            & (u + v <= 1 + 16 * eps)
            & (t > 0)
            & (t <= lower)
            & (t <= upper)
        )

        candidates = torch.where(eligible, t, float("inf"))
        distance, slot = candidates.min(dim=1)

        selected = (
            torch.isfinite(distance)
            & (distance < hit.distance[rows].double())
        )

        if not bool(selected.any()):
            return hit

        local = selected.nonzero(as_tuple=True)[0]
        destination = rows[selected]
        chosen = slot[selected]
        face = adjacent[local, chosen]

        weights = torch.stack(
            (1 - u - v, u, v), dim=-1
        )[local, chosen].clamp(0, 1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        indices = hit.face_index.clone()
        lengths = hit.distance.clone()
        barycentric = hit.barycentric.clone()
        normals = hit.normal.clone()
        matched = hit.hit.clone()

        indices[destination] = face
        lengths[destination] = distance[selected].to(lengths.dtype)
        barycentric[destination] = weights.to(barycentric.dtype)
        normals[destination] = self.normals[face]
        matched[destination] = True

        return type(hit)(
            indices, lengths, barycentric, normals, matched
        )
