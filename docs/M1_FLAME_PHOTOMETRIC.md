# FLAME photometric fitting with gffx + PyTorch

## Completed B1 — 15 October 2026

A complete actual FLAME photograph fit now exists. B1 preserves HavenFeng83f84b804938af6fe7472e492ec997e38552069d at224px,200pose/camera+800joint updates,all236controls,complete traces and durable checkpoints. Both fixed pose-aware recovery controls pass their prospective criteria. Photograph likeness remains unacceptable; schedule completion is not convergence or unique identity recovery.

Executed wheel identities: gffit cd385f09e4f6c01da1bf8b4a0cb18e4009e03ae9af6ba927056eae38bf42b95f; gffx9ac7d34861cd60ad4b4da3ae0210808cb863f1d6978f7e6086f7b3a5aed9a655. The manifest's fitting_baseline_b1 index binds sources,assets,versioned68estimated inputs/mask,config,validation andoutputs. No licensed assets are bundled.

B1 uses public CPUfloat64 source-compatible hard raster,linear texture/SH,deliberate interpolated-normal/coverage stop-gradients,source area-epsilon/depth conventions andexplicit224workspace. Three fixed-state/all236dataVJP comparisons pass; full original-stack parity remains unverified. This is distinct from the original unit-normal/full-normal-derivative example below and filteredSH136/privateCUDA. No qualification transfers among them.

Group4 and M1 are complete for the explicitly adopted development-readiness boundary; finite Group5 technical presentation is accepted. See M1_DEVELOPMENT_READINESS_CLOSEOUT.md. This does not close independent filteredSH136 qualification, full-stack parity or photograph likeness. Campaign/schedulers stay paused indefinitely; no numerical or next-method execution follows.

## Historical 3 October 2026 example — preserved

All pending/no-fit statements below describe the earlier profile and checkpoint, not current B1. Failures,limits andartifacts remain preserved.


This is a Group 4 experiment, not the gffit application or a completed real-image
fit. `tools/flame_photometric.py` composes the installed gffx wheel. No PyTorch3D,
DECA renderer, Chumpy runtime or external FLAME implementation is imported.
The input model is data; restricted assets are never bundled here.

## Reference and rendering profile

Historical reference: https://github.com/HavenFeng/photometric_optimization at
`83f84b804938af6fe7472e492ec997e38552069d`. The exact seven inspected source files
and their hashes are preserved in the internal `group4-flame-20261003` evidence
directory. They are evidence, not executed code or runtime dependencies.

The baseline uses 100 shape, 50 expression, six global/jaw pose, three camera,
50 texture and 27 lighting scalars, 200 initial landmark iterations and 800
joint iterations at224x224. This workload has NOT been admitted or launched.

Our explicit initial profile is `gffx-unit-normal-hard-sh-v1`: hard rasterization,
bilinear texture sampling, spherical-harmonic illumination, normalized shading
normals and full normal derivatives. It retains the reference's camera axis
conventions, texture/SH coefficient meanings and objective normalization.
It does not claim upstream renderer parity: upstream detaches visibility and
shading normals, uses nonunit interpolated shading normals, float32 and its own
raster/tie rules. The present profile is neither the default opaque Principled
renderer nor a replacement for Groups1-3. Those qualifications remain intact.

## Operation crosswalk

| Reference operation | This implementation |
|---|---|
| Shape/expression blend and pose corrective offsets | `gffx.torch.mesh.blend_shapes` |
| Joint regressor | PyTorch linear model-data contraction |
| Axis-angle rotation | `gffx.torch.transforms.rodrigues` |
| Local skeleton transforms and inverse bind | PyTorch assembly, `skeleton.forward_kinematics` |
| Skinning | `mesh.skin_lbs` |
| Static/dynamic landmark embedding | Discrete pose-bin lookup, `mesh.evaluate_barycentric` |
| Scaled orthographic camera | `transforms.transform_points`; explicit reference axis conversion |
| Texture PCA | PyTorch linear model-data contraction |
| Rasterization and attributes | `render.rasterize`, `render.interpolate` |
| Texture sampling | `render.texture_pyramid`, `render.texture` |
| Shading normals and SH | `mesh.vertex_normals`, `light.spherical_harmonics` with coefficient conversion |
| Objective/optimizer | PyTorch masked L1, landmark norms, parameter priors and bounded Adam |

## Input contract

`FlameData` contains float tensors with common device/dtype:

- template `[V,3]`, concatenated shape/expression basis `[K,V,3]`;
- pose basis `[36,V,3]` for FLAME's five joints;
- joint regressor `[5,V]`, skin weights `[V,5]`;
- ordered parents `[5]` and triangles `[F,3]` as int32.

`FittingInputs` adds face-corner UV `[F,3,2]` in **v-down image-row** convention,
RGB texture mean `[H,W,3]`, basis `[T,H,W,3]`, static landmark triangle indices
and barycentric weights, and optional79-bin dynamic contour tables. Shape and
expression coefficients are concatenated in that order. Model conversion must
select the correct expression offset for the actual source version, not assume
it follows the selected shape component count in the source file.

`evaluate(inputs, parameters, size=...)` consumes six named groups: `shape`,
`expression`, `pose` (six global/jaw values), `camera` (scale,tx,ty), `texture`,
`lighting` (`[9,3]` in upstream ordering). It returns vertices, projected
landmarks, albedo, colour, mask and depth. Landmark coordinates are normalized
image coordinates with y-down. Depth is a fixed monotone mapping for the bounded
raster profile, not a metric-depth observation.

The caller must convert texture byte scale, channel order and UV orientation
explicitly. No pickle or asset download occurs in this module. The hash-pinned converter in `tools/flame_assets.py` emits numeric NPY files
under the mapped internal recovery/output space; it never modifies the originals.

## Qualification completed in this batch

Independent synthetic controls cover shape-dependent joints, nonzero pose
correctives, inverse bind, embedded landmarks, camera forward/gradients, texture
coefficients, SH conversion, objective normalization and invalid inputs.
An affine-texture image has a closed independent pixel/camera derivative target.
Complete image geometry/texture/light derivatives are checked at two finite-
difference scales. A synthetic five-joint chain checks all six parameter groups
before a fresh finite descent. A64-step three-channel texture recovery has an
independent target. These are small controls, not a real FLAME fit or a proof of
general convergence or unique recovery.

The initial exact-byte diagnostic failed in SH matrix contraction and the texture
coefficient reduction. Both are repaired using explicit products/reduction order.
A fresh installed wheel now passes the complete float64 fixture's output and all
six parameter-group gradient byte comparisons, plus isolated SH first-VJP checks
for shared/distinct batches and empty points. Failed runs remain preserved.
This evidence does not establish parity at actual FLAME sizes or across contour
transitions. The bounded optimizer currently uses Adam without weight decay;
matching the historical 1e-4 weight decay, staged schedule and resumable state
is still pending.

Per-stage optimization is capped at64iterations and90seconds and must run under
the existing120second worker/180second watchdog, one CPU thread, CPU1GiB/CUDA2GiB
host and64MiB active GPU limits. The function's finite mesh/texture/extent guards
are rejection boundaries, not admission evidence for every value inside them.
Calling another stage does not renew an exhausted experiment's allowance.

## Remaining work before Group4 can close

1. Assets located and hash/header audited at `D:/models/flame`:
   `2020/generic_model.pkl` and `FLAME_texture.npz` are the initial candidates.
   Original files remain unchanged; no symlink is required. The source template
   and original landmark embedding are in the mapped `flame_fitting_support`
   directory (`reference-assets1`), pinned to the same upstream commit.
   See `assets-found2.json` for hashes, sizes and archive CRC checks. The texture
   basis is `[512,512,3,200]` float64; bounded conversion must stream/select the
   intended components rather than load the entire basis into a1GiBworker.
   The105point MediaPipe embedding and model-region masks are separate resources,
   not substitutes for the original68landmark embedding or a photograph mask.
   The user selected the homepage portrait from danzelserrano.com. Its original
   1024x1024RGB PNG is stored unchanged under the mapped `datasets` directory
   (`target-danzel1/enhancedicon.png`), with source URL and hash in provenance.json.
   Draft68landmarks, a conservative43844-pixel skin mask and an explicit500x500crop
   transform are saved in portrait-annotations1. They are model-authored estimates,
   not measured ground truth. The draft is now available in the testing dashboard at
   `/testing/?page=portrait-review`, with overlays, numbered points, zoom and
   full-photo/crop controls. Browser rendering and phone layout are verified;
   user annotation acceptance and fitting acceptance remain pending.
2. Conversion is qualified for these hash-pinned files; static real-model deformation,
   camera and full68landmark derivatives are independently checked. Qualify dynamic
   contour-bin transitions and the complete real-size image/texture path next.
3. Reconcile the historical-compatibility profile with the desired modern gffx
   appearance/derivative profile. Initial actual-model geometry/landmark/first-gradient byte checks now pass;
   full rendered-image/texture parity and the intended fitting profile still need
   qualification. Synthetic parity alone does not establish real-image fitting.
4. Admit cost/memory for actual parameter counts, texture sizes and observations,
   then execute a prospectively bounded staged fit with preserved optimizer state,
   full source/input/wheel identities and failures. No1000-step campaign is implied.
5. Inspect actual recovered images/parameters, independent checks and novel views
   where available; publish explanatory dashboard evidence and reconcile M1.

## Real-asset conversion and initial geometry checkpoint

`conversion-real3.json` records16 numeric arrays, all hashed after writing to the
new `converted-assets1` directory.100shape and50expression components are selected
with expression offset300. Texture conversion selects50components at256x256 using
the upstream nearest-neighbour convention, BGR-to-RGB and division by255 in float64.
The largest streamed texture row is2457600B; complete conversion uses224100352B
sampled host memory and4.4494208supervised seconds. The selected texture basis alone
is78643200B: it must stay on host or use a qualified bounded path, not be moved
wholesale into a64MiB GPU allowance. No dimension reduction is authorized implicitly.

The original landmark object file contains old Torch CPU storage envelopes.
The converter parses their inert numeric data without Torch's nested pickle loader,
accepts only known contiguous layouts, and removes the explicit singleton full-table
batch axis. Barycentric partition checks sum stored values in float64 against the
unchanged1e-7 criterion; no weights are normalized or clamped. All22conversion controls
and the repository teaching check pass (`conversion-qualified1.json`). Earlier
missing-module, unsupported-format and float32-validation failures remain preserved.
Model faces and texture corner indices match the pinned OBJ exactly, as do UV values.
The OBJ vertex positions differ from the2020model; only topology and UV data are used
from that file. Its template positions never replace the selected model template.

`real-geometry-cpu2.json` and `real-geometry-cuda2.json` check one nonzero150coefficient,
six-pose,three-camera case against independent NumPy deformation/projection equations
and six directional differences (three joint parameter families at two step sizes).
Vertices, all68projected landmarks and all tested first-gradient arrays match exact
CPU/CUDA bytes. Maximum independent forward error is5.551115123125783e-16; maximum
directional discrepancy is7.636868568083877e-13. CUDA peak is49517056B. This is initial
geometry qualification, not all poses, dynamic contour transitions, rendered pixels,
real-subject fitting convergence or a final showcase.

The first real-model byte comparison failed despite numerical accuracy. A diagnostic
isolated device-dependent model matrix products. Fixed pair-tree products and first
adjoints now qualify at actual5023-vertex/five-joint sizes and retain the complete
synthetic fitting-chain tests (`model-fixed-qualified1.json`:24passes including guides).
The installed gffx wheel is unchanged; these repairs are in the host example.

No real-subject fit, new dependency, Git publication, history deletion or quota increase
has occurred. Group4 and M1 remain unfinished.

## Reviewing the portrait

Open `https://danziepc.tail4c8306.ts.net:8444/testing/?page=portrait-review`, or
choose **Portrait input review** in the dashboard sidebar. The M1 overview links
there too. Toggle yellow landmarks, point numbers and green skin regions; switch
between the face crop and full photograph, or zoom in. Use point numbers when
requesting corrections. Viewing the page does not approve annotations or start fitting.

This uses numeric review data and the original hash-addressed photo in the existing
internal dashboard store. It does not serve the earlier local HTML file, expose a
filesystem browser, add a service, or change browser security settings. The prior
localfile navigation refusal is preserved as history. Eight integrity/refusal
controls and the repository teaching check pass; live browser inspection verifies
the page and390x844phone controls. Observations remain drafts pending user review.
