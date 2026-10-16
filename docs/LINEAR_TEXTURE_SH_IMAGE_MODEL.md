# Finite linear-texture / SH image composition

`gffx.torch.render.linear_texture_sh` is an explicit image model. It accepts finite signed linear texture values and unclamped polynomial spherical-harmonic shading. It does not change the classic or principled physical-material domains. PNG display clipping is a caller concern and must not become objective clipping.

The ordinary operation uses exact-affine barycentrics and the public scene/camera visibility path. Its defaults and normal differentiation remain unchanged.

The optional `affine_denominator_offset` and `visibility_depth` arguments support image models whose barycentrics use a signed NDC triangle-area offset. The caller supplies one finite nonnegative visibility depth per vertex and one parallel camera. For signed area A and offset e, the operation uses q=A/(A+e), rejects abs(A)<=e, selects depth q*sum(lambda*z), and returns q*lambda. UVs interpolate in the centred [-1,1] image-grid convention before conversion back to [0,1]. The source profile is an explicit fitting composition: e=1e-8 and image depth=10-scale*world_z. It does not redefine physical camera depth.

This opt-in path uses public rasterize/interpolate operations and an explicit eight-fragment list to exclude exact zero-edge barycentrics without hiding a valid face behind an excluded edge. If all eight fragments are occupied by excluded edges, it refuses; it does not silently invent background. This is a bounded path, not unbounded-overlap qualification. Ordinary rasterization retains its existing inclusive-edge contract.

Workspace planning retains the explicit CPU allowance contract (default64MiB, caller opt-in at most512MiB). Duplicated geometry and fragment-list storage are charged in addition to ordinary planning. Invalid inputs and workspace excess refuse before the corresponding allocation.

`stop_normal_gradient=True` detaches the complete interpolated normal image. It does not renormalize that image, clamp SH, or add visibility derivatives. The default keeps normal differentiation available to other image-model callers. UV interpolation and texture/light derivatives remain active. Hard owners and binary masks are discrete.

The 2026-10-15 fixed-state validation independently compares NumPy source equations and a separate Torch primitive VJP with this public operation. It includes strict shared edges, equal-depth ownership, zero-border bilinear texture sampling, signed shading and normal-stop-gradient checks. Evidence is in the registered gffx/gffit conformance/refit packet. Three fixed states and smooth-branch checks do not establish complete original CUDA-stack parity or renderer-wide qualification.
