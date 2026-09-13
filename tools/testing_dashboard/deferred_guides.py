"""Authored explanations for deferred checks; not acceptance results."""
GROUPS=(('python/test_deferred',
    'Small synthetic CPU controls. These do not establish full-image Principled memory, CUDA parity, photographic quality or consumer fitting acceptance.',
    '''
deferred_facade_namespace_is_lazy|Inspecting the renderer namespace should not load an autodiff framework.|Use a fresh interpreter to access the callable facade and unknown import-probe names; verify torch remains unloaded and repeated imports preserve identity.
deferred_public_facades_delegate_without_mutating_inputs|Users should receive the same renderer through import gffx without losing namespace access or changing their inputs.|Compare base and bound callable modules with the implementation, check namespace introspection and shared descriptor identity, and verify input tensors remain unchanged.
deferred_contract_analytic_quad|A simple lit square should have exactly predictable colour and geometry.|Compare a front-facing quad with albedo divided by pi and pixel rays on its known plane; verify coverage, normals and depth.
deferred_empty_background_and_options|An empty scene should be safe and invalid options should not be silently accepted.|Check background RGB, zero coverage, infinite background depth and empty visibility; reject malformed options even without geometry.
deferred_triangle_perspective_plane|A tilted triangle must not warp its attributes through affine image interpolation.|Intersect independent camera rays with a known sloping plane and compare perspective-correct rendered positions.
deferred_albedo_gradient_and_mesh_convenience|Changing surface colour must produce the expected derivative and convenience calls must not change rendering.|Differentiate the analytic quad and compare mesh convenience against an explicit Scene using identical material and light bindings.
deferred_hard_mask_has_zero_geometric_derivative|Hard visibility cannot provide a smooth silhouette-fitting gradient.|Differentiate the hard coverage sum and require finite exactly zero derivatives to all vertices.
deferred_soft_mask_has_nonzero_derivative|The declared soft silhouette approximation should provide a useful local derivative.|Differentiate a bounded signed-distance coverage ramp on a small quad and require finite nonzero vertex derivatives.
deferred_interpolate_rejects_unstructured_landmarks|Grid interpolation must not be advertised as a landmark-point operation.|Supply a rank-one face list to the existing native interpolation wrapper and require its documented shape rejection.
deferred_budget_preflight_and_duplicate_outputs|Oversized work and ambiguous output requests must fail before rendering.|Reject an insufficient workspace allowance and duplicate requested output names.
deferred_supersampling_is_manual_box_and_preserves_ids|Supersampling must smooth colour without inventing averaged surface identities.|Compare against manual double-resolution rendering and box reconstruction; require geometric channels equal centre-sampled output.
deferred_attribute_texture_source_and_map_gradient|Image textures should reach rendered colour and remain fit parameters.|Sample an explicit two-by-two map through the native sampler and compare colour; differentiate the render back into every source texel.
deferred_rig_matches_separate_views_and_shared_gradient|Multiple cameras must share geometry without mixing views or breaking optimization.|Compare a two-camera batch with separate calls and require one weighted loss to reach shared vertices and camera intrinsics.
deferred_visibility_pixel_centre_and_occlusion|Visibility should separate front surface samples from occluded, behind-camera and offscreen samples.|Place exact pixel-centre depth probes around a known front quad and assert the declared sampled-depth classifications.
deferred_soft_boundary_extension_keeps_albedo_in_domain|Soft pixels outside a triangle must not extrapolate valid albedo to invalid negative values.|Render extreme vertex colours with soft coverage and verify the explicit barycentric boundary extension remains finite and bounded.
'''),('python/test_deferred_lessons',
    'Original worked-example checks, not renderer or device acceptance.',
    '''
deferred_lesson_perspective_example_matches_rational_arithmetic|The textbook should teach the correct perspective weights.|Compare its scalar example against exact Fraction arithmetic, including the explicit four-sevenths, two-sevenths and one-seventh answer.
deferred_lesson_premultiplied_example_and_boundaries|The textbook should explain transparent-edge filtering without overstating rendering support.|Recompute red-over-white from premultiplied values and verify key numerical and implementation limitations remain stated.
'''),('python/test_deferred_edges',
    'Failure-first scalar and empty-source validation checks; not full shading qualification.',
    '''
deferred_soft_radius_squared_must_be_representable|A finite softness value must not create NaN coverage when squared.|Reject extreme positive radii whose squared value underflows or overflows before rasterization.
deferred_empty_scene_still_validates_source_structure|Empty geometry must not conceal malformed attribute or texture bindings.|Reject invalid source names, scalar flags, texture dimensions and offset types without evaluating an expensive material.
deferred_source_parameters_reject_mixed_dtype_even_when_unused|A dormant material must still respect the scene precision contract.|Attach a float32 colour to a float64 scene and require an explicit type error even when only coverage is requested.
'''),('python/test_deferred_recompute',
    'Planned first-order recomputation checks on tiny CPU float64 scenes. No higher-order, large-image or GPU acceptance is implied.',
    '''
deferred_principled_six_pixels_fit_without_retaining_all_point_graphs|A small image should preserve roughness gradients without retaining every expensive shader graph.|Render six pixels through bounded Principled blocks and require finite colour and a nonzero roughness derivative; the previous whole-image guard rejects this baseline.
deferred_recompute_matches_unchunked_values_and_every_fitted_gradient|Recomputing a shader during backward must not change its mathematics.|Compare exact values and all normals, view, light, albedo, roughness and F0 partials with an independently called two-point public material.
deferred_recompute_saves_inputs_not_material_intermediate_graph|Small forward calls alone do not guarantee bounded retained memory.|Inspect saved tensor storage identities and require only the explicit original inputs, then verify backward still reaches roughness.
deferred_recompute_roughness_derivative_matches_two_finite_differences|A reconnected roughness gradient must agree with actual value changes.|Compare the analytic derivative with central differences at two step sizes away from resource boundaries.
'''),('python/test_deferred_light_edges',
    'Tiny invalid-sample composition control. Coincidence is undefined physically; its zero value and derivative are the declared invalid-branch convention, not a dark lamp.',
    '''
deferred_point_light_coincidence_is_finite_zero_with_zero_gradient|A point light exactly on the sampled surface should contribute the defined zero instead of breaking shading.|Place one point light at the sole pixel surface point and require exact zero colour and position derivatives.
deferred_diffuse_sh_respects_one_sided_view_hemisphere|Diffuse environment lighting must not illuminate the back of an explicitly one-sided material.|Disable geometric culling on reversed faces and require zero RGB and environment-coefficient derivatives from the unviewable material side.
deferred_two_sided_sh_rotates_to_the_viewed_normal|An explicitly two-sided surface must see the environment from its flipped shading normal.|Use a directional SH coefficient on reversed faces and compare the result with an independent positive-z diffuse environment value.
'''),('python/test_deferred_integration',
    'Tiny analytic CPU image controls. Independent algebra tolerances do not replace reference-mode bit identity, full-image memory qualification or visual acceptance.',
    '''
deferred_matches_manual_native_chain_bit_exactly|The high-level renderer should preserve the exact mathematics of its public building blocks.|Manually compose projection, rasterization, perspective interpolation, normals, BRDF and lighting without integrator helpers; compare every requested buffer bit-exactly.
deferred_point_lights_match_independent_inverse_square_image_and_partials|A lit image must obey distance falloff and expose useful light-fitting gradients.|Independently intersect pixel rays with a known plane, evaluate two inverse-square lamps and compare RGB plus position and intensity derivatives.
deferred_named_camera_and_point_light_hierarchy_matches_world_configuration|Parented cameras and lights should behave like their explicit world-space equivalents.|Translate a complete scene hierarchy, compare every requested buffer bit-exactly with manually translated geometry, light and camera.
deferred_two_sided_shading_flips_brdf_and_cosine_together|A back-facing surface should light correctly only when two-sided shading is explicitly enabled.|Reverse the quad winding, disable culling, require black one-sided output and analytic diffuse RGB with two-sided normals.
deferred_diffuse_sh_coefficients_match_independent_basis_and_gradient|Diffuse environment coefficients should have predictable image values and derivatives.|Compute the real spherical-harmonic basis at the known plane normal independently and compare colour and every coefficient derivative.
deferred_multiple_materials_keep_pixel_identity_and_separate_gradients|Neighbouring objects must retain their own materials without mixing fit parameters.|Render two half-image quads, check exact material IDs, analytic RGB and separate eight-pixel albedo derivatives.
'''),('python/test_deferred_resources',
    'Pre-allocation refusal controls use spies, not large allocations. Passing them alone does not qualify large-image memory or timing.',
    '''
deferred_resource_refusal_precedes_scene_flatten|A refused workload should not first allocate its packed geometry.|Request an impossibly small byte allowance and spy on scene flattening to prove rejection occurs before packing.
deferred_resource_estimate_counts_light_packets_before_raster|Pixel count alone cannot bound memory when many light samples are used.|Supply a tiny one-pixel scene with many explicit light samples and require budget refusal before raster allocation.
deferred_resource_components_sum_and_scale_with_samples_and_supersampling|The workload estimate must respond to the actual light and image dimensions.|Inspect estimate components, require their exact sum, fivefold sample storage for five lights and fourfold sample storage for double image dimensions.
deferred_resource_principled_scratch_is_added_to_live_storage|Shader and renderer memory allowances must be added instead of treated independently.|Recompute the admitted four-point material scratch formula and require that it is included alongside live renderer storage.
deferred_resource_buffer_estimate_respects_actual_reuse|A conservative estimate should not charge twice for the same reused geometric buffer.|Compare ordinary, supersampled colour-only and supersampled geometric requests and verify exactly when a separate centre-sample buffer is counted.
deferred_resource_lighting_counts_disjoint_material_pixels_once|Adding material identities should not pretend that each pixel is shaded by every material.|Compare one and two material tables and require identical pixel-light storage, since one retained fragment selects only one material per pixel.
deferred_resource_light_only_fit_counts_backward_and_borrowed_storage|Fitting only a light must still reserve derivative memory even when geometry and material are fixed.|Compare otherwise identical static and fitted irradiance bindings and require the backward flag and larger input and lighting allowances.
deferred_resource_immutable_budget_has_no_duplicate_adjoint_storage|An immutable lighting table should be counted once without inventing derivatives of the table itself.|Compare static and fitted material plans while retaining the full separately charged query/backward scratch allowance.
deferred_resource_alias_in_ordinary_binding_keeps_derivative_allowance|The constant-resource exception must not hide the same storage used as an ordinary callback parameter.|Supply an explicit alias through a separate ordinary material binding and require its original derivative allowance to remain.
'''),('python/test_deferred_profile',
    'Prospective serialized 8x8 built-in CPU float64 profiles. Execution requires planner controls to pass first; no larger-image, CUDA or arbitrary-callback admission follows.',
    '''
deferred_8x8_principled_forward_profile|The bounded shader should progress from a few points to a complete small image.|Preflight and render an eight-by-eight opaque quad with an explicit admitted Principled resource and require finite positive RGB and full coverage.
deferred_8x8_principled_backward_profile|A complete small image should still expose a material-fitting derivative.|Preflight the gradient-bearing workload, render sixty-four pixels and differentiate the summed image into shared roughness under the supervised worker limits.
'''),('python/test_deferred_gradients',
    'Tiny central differences away from declared visibility, coverage and sampler discontinuities. These do not imply derivatives through discrete topology or selection changes.',
    '''
deferred_soft_silhouette_gradient_matches_two_steps_in_stable_selection|The soft silhouette gradient should predict actual local changes, not merely be nonzero.|Perturb a nonsymmetric triangle in a fixed direction and compare the rendered coverage derivative with central differences at two step sizes.
deferred_authored_normal_photometric_gradient_matches_two_steps|Fitting authored normals should change the actual rendered lighting correctly.|Perturb interpolated surface normals under an oblique light and compare the image-loss derivative with two independent central differences.
deferred_uv_photometric_gradient_matches_two_steps|Fitting texture coordinates should retain the sampler and rendering derivatives.|Move UV coordinates within a smooth bilinear cell and compare the full photometric gradient with two central-difference steps.
deferred_masked_rig_photometry_reaches_shared_geometry_and_every_camera|Two image views should fit shared geometry and material together while updating each camera separately.|Compare a masked two-camera photometric objective with separate views, check geometry and camera derivatives with two finite-difference steps, then require a fresh joint gradient update to lower the loss.
deferred_masked_sh_photometric_gradient_matches_two_steps|A masked fitting objective should differentiate into diffuse environment coefficients correctly.|Perturb low-band spherical-harmonic coefficients and compare the actual masked image-loss change with two central-difference steps.
deferred_masked_albedo_source_gradient_matches_two_steps|Both vertex colours and texture pixels should remain fitted inputs through the complete masked renderer.|Compare vertex-albedo and freshly rebuilt texture-pyramid derivatives with central differences at two step sizes under a fixed photometric mask.
'''),('python/test_deferred_demo',
    'Original built-in studio demonstration. Geometry controls and a32pixel smoke render do not establish visual quality, large-image performance or shadows/indirect illumination.',
    '''
deferred_studio_geometry_is_nondegenerate_and_sphere_winding_is_outward|The demonstration should begin with valid smooth geometry rather than misleading shading artifacts.|Check sphere and ribbon topology, nonzero triangle areas, outward sphere winding and unit authored normals.
deferred_studio_32_smoke_uses_actual_public_renderer_frames|The proposed studio demonstration must show actual gffx rendering rather than substitute display shading.|Render three fixed camera views of a metallic orb and matte ribbon through the public built-in renderer and require finite changing images.
'''),('python/test_deferred_field_roles',
    'Three-pixel shape-collision controls for explicit built-in material roles. Custom callback parameter interpretation remains its own contract.',
    '''
deferred_three_pixel_rgb_field_rejects_scalar_axis_removal|Three grey point values must never be silently reinterpreted as one constant RGB colour.|Bind a three-pixel specular field with scalar-axis removal and require an actionable channel-axis error before shading.
deferred_three_pixel_grey_field_retains_channel_axis|A legal grey field should vary across pixels while remaining achromatic.|Keep the trailing one-channel axis for specular samples and require equal RGB channels but different values at the left and right pixels.
'''),('packaging/test_deferred_preflight',
    'Standard-library arithmetic only. These are conservative charged planner terms, not measured allocations or evidence that any full render is admitted.',
    '''
deferred_preflight_refuses_256_buffer_without_allocating|Large planned image sizes should be ruled out safely when their buffer term alone exceeds the envelope.|Calculate the float64 256-square buffer term with integer arithmetic and require a refusal without importing the renderer.
deferred_preflight_small_bound_never_claims_admission|A partial conservative estimate must not be confused with a complete memory plan.|Check that the 128-square target still requires full preflight including the omitted geometry, lighting and shader resources.
deferred_preflight_counts_only_separate_geometry_buffers|Soft and supersampled rendering should charge additional geometry buffers only when they are actually distinct.|Compare hard, soft and supersampled arithmetic with and without requested geometric outputs.
'''),('packaging/test_deferred_cached',
    'Transport tests use synthetic byte fixtures, not renderer outputs. Cached demonstration publication never substitutes for numerical acceptance or visual review.',
    '''
deferred_cached_verifies_finite_frames_without_torch|Sending already rendered demonstration images should not rerun an expensive renderer.|Verify six bounded cached images and pinned source, wheel and report identities through the dependency-free transport.
deferred_cached_rejects_changed_pixels_before_publication|A changed image must not be presented under an old rendering claim.|Modify one cached raw image and require verification to fail before any dashboard call.
deferred_cached_rejects_unpinned_report|The rendering report itself must be pinned, not merely the wheel named inside it.|Replace the report after its digest was selected and require an identity error.
deferred_cached_uses_only_allowed_artifact_provenance|A demo should publish provenance in the dashboard's accepted schema.|Validate the report digest as artifact provenance without inserting unsupported metadata fields.
'''),('packaging/test_deferred_head_tooling',
    'Allocation-free profile metadata controls only; passing them does not validate a mesh, render an image or qualify DF04.',
    '''
deferred_head_profile_metadata_is_explicit|The fixed-head comparison should use the same declared camera, light and material profiles as its reference.|Validate the two named materials against explicit128-square camera and single-light metadata without allocating tensors.
deferred_head_profile_rejects_changed_camera_before_geometry|A changed camera must not silently reuse an unrelated reference image.|Change the declared resolution and require a profile error before mesh construction.
deferred_head_profile_rejects_other_material_or_light|Different illumination or shader semantics require a different reference claim.|Reject an undeclared material and changed irradiance rather than comparing them to the fixed profile.
'''),('packaging/test_deferred_head_cost',
    'Metadata arithmetic uses synthetic calibration records. Forecasts are not measured head render times, hard runtime bounds or replacement for process supervision.',
    '''
deferred_head_cost_uses_slower_per_pixel_rate_and_margin|The full-head attempt should be informed by smaller measured renderer workloads.|Use the slower16/32 per-pixel rate, double it for the pinned6536 covered pixels and reserve30seconds for other overhead.
deferred_head_cost_rejects_changed_artifact_or_backward_profile|Timing from the wrong build or workload must not authorize a different render.|Require matching artifact and driver identities plus supervised forward-only CPU calibration profiles.
deferred_head_cost_refuses_forecast_over_cooperative_deadline|A predictably overlong head render should be refused before it begins.|Require a prospective runtime above120seconds to stop during metadata validation.
deferred_head_cost_requires_matching_explicit_point_partition|A slower one-point schedule must not borrow timings from four-point shading calls.|Require both measured calibration reports to state the same point partition as the candidate and reject mismatched or missing partition metadata.
'''),('python/test_deferred_storage',
    'Pointer and shape audit of an admitted two-point CPU backward. Saved storage is not total transient memory; unchanged process supervision remains authoritative.',
    '''
deferred_budget_constants_stay_borrowed_through_recomputed_backward|Fixed lighting-resource tables should remain immutable borrowed inputs during fitting.|Record only saved-storage pointers and shapes, verify unchanged table versions and absent table gradients, and retain the complete material scratch allowance.
'''),('python/test_deferred_cuda',
    'Prospective tiny exact CPU/CUDA qualification, not a large-image or realtime claim. Requires a supervised CUDA worker; any numerical mismatch remains a failure, never a tolerance downgrade.',
    '''
deferred_cuda_hard_geometry_and_identity_match_cpu|GPU rendering should preserve the CPU geometry and object-selection contract.|Compare every hard geometric output and visibility on the same one-pixel triangle in both supported floating precisions.
deferred_cuda_classic_colour_and_matched_albedo_vjp|The simple GPU shader should preserve both pixels and fitting derivatives.|Use identical RGB cotangents and require exact CPU/CUDA classic colour and nonzero albedo derivatives.
deferred_cuda_soft_silhouette_and_vertex_vjp_match_cpu|Soft outlines should differentiate consistently on CPU and GPU.|Render the same asymmetric stable one-pixel silhouette and compare coverage plus every vertex derivative exactly.
deferred_cuda_principled_same_resource_table_and_fitted_vjps|The admitted GPU material resource should describe the same model and derivatives as CPU.|Verify equal mathematical table payloads from the separately named resources, then compare colour and matched albedo, roughness and reflectance VJPs exactly.
'''),('python/test_deferred_forward_blocks',
    'Tiny forward-only partition controls. Exact pixel equality does not qualify a large image; backward partitioning and all process/workspace caps remain unchanged.',
    '''
deferred_forward_planner_selects_largest_fitting_block|A forward render should use the largest safe shading block instead of refusing solely because four points do not fit.|Compute exact whole-workload limits for four, two and one points and require the planner to select the largest fitting partition while retaining every other charged component.
deferred_forward_one_point_refusal_precedes_flatten|Even the smallest shading block must not bypass the complete memory allowance.|Set the limit one byte below the one-point plan and spy on scene flattening to prove refusal occurs before packed geometry allocation.
deferred_backward_never_shrinks_its_four_point_partition|Changing forward memory scheduling must not change the established fitted-gradient accumulation order.|Require a fitted workload to retain four-point blocks and refuse a smaller allowance; verify that an explicit no-grad context is genuinely forward-only.
deferred_forward_partition_values_are_bit_identical|Smaller blocks should preserve the exact material values at each point.|Compare one-, two- and four-point evaluation against a direct four-point material call with varying colour, roughness and reflectance fields, including explicit singleton batches and noncontiguous field storage.
deferred_forward_runtime_uses_the_admitted_partition|A smaller planner estimate is safe only if the renderer actually uses its chosen block size.|Record real material-call point counts, require every call to respect the chosen limit and compare complete tiny rendered images exactly.
deferred_forward_partition_keeps_point_sample_limit|Point-block adaptation must also preserve the independent light-sample constraint.|Use a harmless material spy with 513 samples and require one-point calls even when the nominal block limit is four.
deferred_forward_partition_handles_empty_and_inactive_work|Memory scheduling should remain safe when there is no visible or illuminated work.|Exercise empty geometry, offscreen geometry, no lights, zero light samples and back-facing illumination; require exact zero colour and the corresponding skipped calls or retained conservative scratch charge.
deferred_backward_partition_preserves_every_matched_cotangent|The forward-only scheduling change must preserve all previously fitted derivatives.|Compare the unchanged four-point-limit path with direct material values and identical cotangents for normals, view, light, albedo, roughness and reflectance; reject a manually forced smaller fitted partition.
'''),('python/test_deferred_byte_identity',
    'CPU-only tests of the CUDA evidence comparator itself, not device qualification. Equal floating values are insufficient when their raw byte patterns differ.',
    '''
deferred_byte_identity_rejects_signed_zero|An exact-byte claim must distinguish positive zero from negative zero.|Show that ordinary tensor equality accepts the pair, confirm different raw bytes and require the evidence comparator to reject it in both floating precisions.
deferred_byte_identity_accepts_equal_scalar_empty_and_strided_values|The evidence comparator should support valid scalar, empty and noncontiguous logical tensor layouts.|Compare matching scalar, empty, boolean and transposed tensors after preserving logical shape and dtype.
deferred_byte_identity_keeps_shape_and_dtype_guards|Equal storage patterns must not conceal a changed tensor contract.|Require an explicit layout failure for changed precision or shape before comparing bytes.
'''),('python/test_deferred_classic_attribution',
    'Matched-operand diagnostic only. Passing means inputs and same-device propagation controls held; any recorded CPU/CUDA byte mismatch remains an unqualified result.',
    '''
deferred_classic_matched_operand_attribution|Locate the first differing operation in a simple lit pixel instead of blaming the full renderer.|Compare identical operands and cotangents through Lambert, classic material, light integration and rendering; record exact output and albedo-gradient bytes in both precisions, verify downstream propagation and separately compare Python-scalar division with an explicit device-tensor denominator.
'''),('python/test_deferred_lambert_division',
    'Tiny exact Lambert division regression. This does not qualify every shader operation or the full renderer device matrix.',
    '''
deferred_lambert_division_matches_device_values_and_vjps|Lambert shading should agree byte-for-byte across devices for the isolated regression inputs.|Compare float32 and float64 public Lambert values and albedo VJPs using identical reflectance and RGB cotangents on CPU and CUDA.
deferred_lambert_division_preserves_cpu_legacy_bytes|Repairing device parity should preserve the measured CPU arithmetic boundary.|Compare the public result and VJP with legacy CPU scalar division on a strided reflectance grid including signed zero and endpoints.
'''),('python/test_deferred_principled_attribution',
    'One-point float64 diagnostic only. A passing attribution control does not accept any recorded CPU/CUDA VJP mismatch or establish a required autograd reduction regrouping.',
    '''
deferred_principled_matched_vjp_attribution|Locate a roughness-gradient discrepancy before changing the physical material or renderer.|Pin all table and input bytes, compare identical RGB cotangents through direct Principled, recomputation and rendering, then record separate channel VJPs and their explicit same-device RGB-order sum.
'''),('python/test_deferred_principled_stages',
    'One-point captured-cotangent diagnostic, not relaxed shader qualification. Its green status covers operand/capture/replay controls; any recorded differing VJP remains unqualified.',
    '''
deferred_principled_captured_stage_attribution|Determine which physical-material component introduces a gradient discrepancy under the real public objective.|Capture inputs, outputs and incoming cotangents at the loss-table, GGX, visibility and Fresnel boundaries, then replay each with identical CPU-recorded cotangent bytes on both devices and record its partial derivatives.
'''))
