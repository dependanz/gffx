# Image operations for fitting and composition

The initial image surface operates on images already held in tensors. It does not decode files,
own editable documents, schedule optimizers or export media. Use `gffx.image` in ordinary Python
and the explicit `gffx.torch.image` backend when examining compilation/export compatibility.
Compilation/export and cross-device bit-identity qualification are not yet established.

Images are float32/float64 `[H,W,4]` or `[B,H,W,4]`, with matching dtype/device and no implicit
batch broadcasting. RGB is linear light; signed/HDR values are allowed. Alpha is in `[0,1]`.
Premultiplied RGBA is the default. The base `import gffx` remains lazy and dependency-free;
calling image functions activates the optional PyTorch adapter.

```python
import gffx
import torch

background = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
red = torch.tensor([[[0.5, 0.0, 0.0, 0.5]]])  # premultiplied
purple = gffx.image.composite([red], background)
# [[[0.5, 0.0, 0.5, 1.0]]]
```

## What each operation means

- `convert_alpha`: explicitly premultiply/unpremultiply RGB; zero-alpha RGB is canonical zero.
- `srgb_to_linear`, `linear_to_srgb`: transfer RGB channel values explicitly, never alpha.
- `apply_mask`: multiply existing alpha, source-space mask and opacity together.
- `composite`: alpha over; the list is bottom-to-top above the supplied RGBA background.
- `reconstruct`: exact-divisible box reconstruction in a fixed summation order, for any channels.
- `affine_matrix`: source-to-destination translation, clockwise radians, scale and explicit pivot.
- `warp_affine`: inverse-map destination pixel centres through nearest/bilinear native sampling.
- `adjust`: multiply by exposure in stops, then add linear-light brightness, preserving alpha.
- `gaussian_blur`: fixed-radius separable blur of premultiplied colour and alpha together.
- `capabilities`: report contracts and unqualified profiles without probing a GPU.

To ingest encoded straight RGBA, decode only its RGB, preserve alpha, then premultiply. To display
or encode a result, explicitly choose the corresponding output conversion. Do not average straight
RGB at transparent edges: invisible colours otherwise leak into visible samples.

Affine pixel centres are `(column,row)`, with y down. The matrix maps source to destination; callers
do not supply a silently inverted matrix. Border choices are `transparent` and `clamp`. Downscaling
with the initial sampler can alias: automatic mip/EWA minification is not implemented. Supersampling
and box reconstruction are explicit alternatives, not a guarantee of perfect minification.

## Gradients and boundaries

Continuous parameters retain their autograd graph. Layer order, dimensions, nearest decisions and
integer blur radius are discrete. Nearest coordinate gradients are zero. Bilinear derivatives at
grid knots follow the selected floor-cell branch; alpha-zero RGB derivatives are zero. Transfer
function thresholds use their linear branch. Finite-difference tests exclude these nonsmooth points
and dedicated tests check their stated conventions. Higher derivatives remain unqualified.
The inverse-grid custom backward is explicitly first-order-only: a request to differentiate
that backward raises rather than returning a partial Hessian from saved detached coordinates.
This guard does not prohibit higher derivatives through otherwise ordinary tensor compositions.
The two colour-transfer functions likewise use a first-order-only shared native VJP. Their
candidate C11/CUDA evaluator uses controlled division, exponent reduction and fixed-count
rational-root arithmetic rather than framework-selected scalar division and `pow`. This is a
rounded implementation, not an exactly rounded general power function; independent accuracy,
full-range classification and CPU/CUDA byte identity must be established on the actual artifact.

Mask values and opacity must lie in `[0,1]`; Gaussian sigma is `[0.1,32]` pixels and required integer
radius is `[0,32]`. Radius zero is a fresh identity with zero sigma derivative. No hidden epsilon
regularizes singular affine transforms or near-zero unpremultiply. Unsupported/domain inputs raise
explicit errors rather than clipping silently.

## Ownership and evidence limits

Inputs are not changed; outputs are fresh and contiguous. Each call builds a fresh graph. There is
no persistent image cache or snapshot logging. Callers retain their own reusable source tensors and
release old results/graphs. Domain checks may synchronize CUDA, so this is not a realtime claim.
Cancellation is checked by the consumer between bounded calls, not by killing an in-flight kernel.

The inverse affine map uses an analytical first-derivative rule with a fixed adjacent-pair
reduction tree, carrying an odd final value unchanged. This avoids framework-selected reduction
orders for matrix gradients. Per output frame of `N=H*W` pixels, this part of the graph retains
two N-value coordinate arrays plus the matrix and determinant; backward forms two coordinate
cotangent arrays and reduces each of six matrix components separately. Pairwise temporary arrays
shrink geometrically; no N-by-six Jacobian is stored. These are implementation storage estimates,
not measured whole-process peaks: source/output RGBA, sampling state, validation temporaries,
allocator pools and caller graphs still count against the worker's measured cap. Calls remain
bounded by their caller; large-frame CUDA and higher derivatives are not qualified by this repair.

Gaussian tap multiplication uses the same fixed tree for the tap-weight derivative. The
forward Gaussian, exponential and normalization are unchanged. When sigma requires gradients,
each tap retains the shifted image needed for that derivative; one product buffer and shrinking
pairwise buffers are temporary during backward. Per-image widths reduce each image separately.
No tap-by-pixel Jacobian is materialized, but all retained shifted images still count toward the
memory cap. This ordering repair is not a general cross-platform exponential guarantee.

The contract and tiny independent tests are in `IMAGE_ACCEPTANCE_V0_1.md` in devbrain and
`tests/python/test_image.py` in the source repository. Passing those tests does not establish
Compositron's real revision/lock/undo adapter, full-frame performance, CUDA reference identity,
artistic quality or release readiness. Consult the recorded exact-artifact results.
