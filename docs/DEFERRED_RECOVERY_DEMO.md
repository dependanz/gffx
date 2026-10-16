# Prospective finite renderer recovery demonstration

Fixed before execution, 14 September 2026. Use the original synthetic head and
pinhole view at64x64, Classic material with specular0.1/exponent2, and the original
single directional light. Fit only the constant RGB albedo from[0.72,0.12,0.25]
to the independently fixed target[0.2,0.4,0.6]. Geometry, normals, camera, light,
specular and exponent remain unchanged. This is a small gffx library example,
not gffit, licensed-model fitting or full-image/all-parameter qualification.

Generate target values with the frozen independent standard-library Classic
formula on native geometry buffers. This independently checks material values;
the separately qualified Decimal silhouette oracle checks image edges. Take20
fresh-graph gradient-descent steps with fixed learning rate12 on mean squared RGB
error over covered pixels. Do not retune the target, rate, count or thresholds.
Require finite in-domain iterates, final MSE below1% of the initial value and
maximum parameter error below0.03. At each iterate independently predict RGB
from the scalar formula's fixed constant/linear coefficients and require the
existing algebra bound atol1e-10/rtol1e-9. Preserve every loss and all failures.

Preflight each renderer configuration before allocation. One CPU thread,1GiB
whole-worker host,64MiB workload allowance,90s local demonstration deadline,
120s cooperative/180s watchdog. No runtime or memory quota increase. Export exact
linear target/start/final arrays plus a finite presentation board only after the
numerical gate. Display uses labelled sRGB and a fixed x20 linear-error scale.
Publication follows visual inspection and verified PNG transport. No inference
of high-resolution throughput, unique general inverse problems or real-time use.
