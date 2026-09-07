# Offline demonstration assets — separate upstream terms

These small fixtures are for the repository's developer demonstrations. They are not
gffx algorithms, not required to import/use gffx, and are excluded from wheel/source
distributions. Do not assume gffx's code license licenses the Stanford data.

## Stanford bunny

Source: Stanford University Computer Graphics Laboratory,
<https://graphics.stanford.edu/data/3Dscanrep/>; downloaded 2026-09-06 from
<https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz>.
`bunny.json` contains the archive's supplied `bun_zipper_res3.ply` coordinates and
triangles, not the full-resolution reconstruction. No hole filling is performed.

The upstream page permits research use and free mirroring/redistribution with
acknowledgement. It restricts commercial use/products for sale without permission.
Its separate bunny/armadillo exception concerns culturally inappropriate animation,
not an explicit waiver of the commercial-use restriction. Obtain permission before
commercial redistribution; this fixture is not advertised as unrestricted open data.

## Utah teapot

Martin Newell's classic teapot, using the patch data distributed in three.js r180:
<https://github.com/mrdoob/three.js/blob/r180/examples/jsm/geometries/TeapotGeometry.js>.
The complete upstream MIT notice is preserved in `three-LICENSE.txt`. No three.js
runtime, npm package or renderer is used. Our stdlib converter evaluates the 32
bicubic patches at six subdivisions, discards zero-area triangles, and rotates to
Y-up. Seams are not welded: this is a display fixture, not a manifold test oracle.

## Reproduction

Download the two exact sources plus
<https://raw.githubusercontent.com/mrdoob/three.js/r180/LICENSE> as
`three-LICENSE.txt` into a scratch build directory. Run
`python -m tools.testing_dashboard.build_classic_assets <directory>`.
The converter verifies pinned source SHA-256 hashes before converting. Each JSON
also carries its source URL, conversion recipe, source digest and license note.
Normal demo publication only reads bundled JSON: no network fetch or new dependency.
