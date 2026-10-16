# Sampled classic SH diffuse visibility

For the explicit zero-specular SH path, radiance is albedo/pi times SH irradiance evaluated at the normalized shading normal. Rasterization owns coverage and depth. A one-sided selected surface emits only where dot(geometric face normal, view)>0. A two-sided material orients the shading normal by the geometric face side; exact geometric horizons emit zero. Smooth shading-normal/view crossings do not clip radiance or flip the illumination frame. Normal derivatives remain connected through SH; side selection is discrete, with no claimed derivative at a geometric horizon.

This is a diffuse irradiance approximation, not a fully occlusion-aware or energy-certified transport model. SH coefficients can be signed; no clamp or second illumination cosine is introduced. Culling remains explicit and independent of two_sided. Standalone classic BRDF and directional/Principled paths retain their documented hemisphere rules. Existing public field validation remains mandatory, including zero-specular admission.

The prior SH path incorrectly borrowed a shading-normal view gate from classic BRDF evaluation. Saved FLAME targets/fits using that old model remain historical evidence. New targets must carry a new renderer identity. This does not retroactively change accepted Principled opaque frame evidence.

Primary context: https://lightfield.stanford.edu/papers/envmap/ describes irradiance as a function of surface normal. https://pbr-book.org/3ed-2018/Materials/BSDFs distinguishes geometric and shading normals and explains dark spots from conflating their hemispheres. These sources motivate the distinction; the exact approximation and boundary conventions above are this renderer's explicit contract.
