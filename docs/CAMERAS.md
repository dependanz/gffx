# Eager pinhole cameras

`gffx.camera` is the first convenience layer over the existing transforms. It uses
PyTorch 2.10+; `gffx.torch.camera` exposes the identical bound class. Plain
`import gffx` does not load PyTorch. This is pre-release source functionality, not
the unrelated old package on PyPI. No JAX, compilation/export or real-time budget
claim is made for this eager object.

```python
import torch
import gffx

camera = gffx.camera.Camera.from_intrinsics(
    fx=500., fy=500., cx=320., cy=240.,
    image_height=480, image_width=640, near=0.1, far=100.,
)
points = torch.tensor([[0., 0., -2.], [0., 0., 2.]])
projected = camera.project(points)
print(projected.pixels)       # image coordinates, x right and y down
print(projected.valid)        # True, True: both divisions are defined

front = camera.project(points, in_front_only=True)
print(front.valid)            # True, False: explicitly reject behind camera
recovered = camera.unproject_points(
    projected.pixels, projected.view_depth, kind="view",
)
```

## Validity is not visibility

The camera looks down its local **negative z axis**. By default, validity uses the
existing perspective divide's `abs(w) > eps` rule. A finite behind-camera point
can be mathematically projected. A point too close to the camera plane is invalid.

`in_front_only=True` adds `view_depth > eps`, using the same epsilon. Rejected
points return false validity and zero pixels, NDC depth and view depth. Their
gradients are zero for finite inputs. Accepted outputs are unchanged. This is a
hard, non-differentiable mask, not a soft visibility function.

Neither mode tests screen bounds, near/far clipping, occlusion or whether a mesh
surface faces the camera. Those remain separate decisions. Always use `valid`:
under the default mapping, invalid zero NDC maps to the image centre, not a special
off-screen pixel sentinel.

## Shapes, depth and device

- Intrinsics are matching `[B]` float32/float64 tensors, or plain numbers for a
  single camera. Plain-number intrinsics can broadcast to a tensor's batch.
- `pose` is `[B,4,4]` **camera-to-world**, defaulting to identity. It must be finite
  and invertible; it is not silently replaced with a pseudo-inverse.
- Points may be `[N,3]` for one camera, `[B,N,3]`, or packed `[P,3]` with monotone
  int32 `point_offsets[B+1]`. `Camera.stack` forms a rig; image size, near and far
  must agree. Empty point sets are accepted.
- Constructor tensors share dtype and device. Projection explicitly casts the
  camera matrix to the points' dtype but never copies between CPU and CUDA.
  Broadcasting/packing may materialize a contiguous temporary; this is not an
  allocation-free API.
- `view_depth` is distance along the camera axis (`-z_view`), not radial range.
  `ndc_depth` is renderer depth, with near at -1 and far at +1.
- `unproject(depth, kind="view" | "ndc", valid=None)` returns `(xyz, valid)` for
  a `[B,H,W]` depth grid. Pixels use centres `(column + 0.5, row + 0.5)`.
  Nonfinite depths and explicitly masked samples produce zero coordinates and
  zero depth gradients. A smaller grid uses top-left coordinates, not implicit
  resizing. `unproject_points` takes explicit pixel coordinates and depths.

Intrinsics, pose, input points and valid depths retain their autodiff graphs.
Image size, near/far, batching and validity masks are static or nondifferentiable.
Distortion, exposure, rolling shutter and lens-stack hooks are currently inert.

## Inspection and finite demonstrations

The development testing application has Camera projection, unprojection,
gradient and validation pages, plus child pages for explicitly reported checks.
The separate `camera/pinhole-demo` page illustrates landmarks, depth reconstruction
and two views of the same geometry. Generate one finite example with:

```powershell
python -m tools.testing_dashboard.camera_demo --address http://127.0.0.1:8791 --run camera-example-unique
```

The producer publishes ten small static artifacts and exits. It does not create
a background stream. Demonstrations and WebGPU display are not acceptance evidence.

`camera/fit-demo` adds a finite four-intrinsic synthetic fit: before/after landmarks,
loss and parameter histories, plus autodiff-versus-finite-difference curves. Geometry
and pose are fixed; this is not full-pose or real-data calibration or the gffit engine.

```powershell
python -m tools.testing_dashboard.camera_fit_demo --address http://127.0.0.1:8791 --run camera-fit-unique
```

Local Phase 3 verification covers projection and dense/scattered unprojection gradients
to intrinsics and pose on CPU/CUDA, inverse depth/pixel derivatives, and independent
two-camera gradients under one backward. Float32/64 device fixtures explicitly check
inherited CUDA composition. These tests do not establish million-point performance budgets.

## Parallel and FLAME weak-perspective projection

`Camera.from_orthographic(fx, fy, cx, cy, image_height, image_width, near, far,
pose=None)` uses positive pixel-per-view-unit scales rather than focal lengths.
For a view-space point `(x,y,z)`, pixels are `(fx*x+cx, -fy*y+cy)` independent
of depth. Clip w is one. View depth is `-z`; NDC depth is
`(2*view_depth-far-near)/(far-near)`. Near/far retain `0 < near < far`.
Pose must be affine and invertible. Zero view depth is mathematically defined;
`in_front_only=True` still excludes points at or behind the camera plane.

`Camera.from_weak_perspective(scale, tx, ty, image_height, image_width, near,
far, pose=None)` implements the normalized image convention
`(scale*(x+tx), -scale*(y+ty))`. Positive scale and translations retain their
PyTorch graphs. It does not introduce a guessed camera distance or change
geometric depth to a legacy fitted depth heuristic. The caller supplies a pose
and clipping planes that contain its geometry.

`projection_kind` is `perspective` or `orthographic`; a rig must be homogeneous.
Sampled deferred rendering preserves this kind through scene camera bindings,
batching and supersampling. Orthographic view vectors are parallel to the
camera's positive local z axis in world space, including specular shading.
Explicit fitting-soft coverage uses the existing approximate profile.
Filtered opaque rendering currently refuses orthographic cameras: its nonlinear
integration and streamed camera snapshots require separate qualification.
This refusal applies to mask-only as well as colour requests, with no fallback.
