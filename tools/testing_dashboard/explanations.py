"""Reviewed reader guides, separate from execution evidence. No source-path reads.

Each row is test name | purpose | method. Parameter variants share the guide,
not their result or identity. The repository inventory test enforces new entries.
"""
GUIDES = {}
CHECK_GUIDES = {}

_ENV_IMAGE_GUIDES = '''
reference_pixel_selection_spans_view_angles_deterministically|The reference controls must include difficult grazing views, not only easy front-facing points.|Sort covered positive-view pixels by viewing cosine and pixel ID, then select eight fixed ranks including both extremes.
classic_constant_environment_closed_form_matches_scalar_quadrature|A complete classic material must include its glossy light response, not just diffuse colour.|Compare independent hemisphere integration with the exact exponent-two classic integral at views where the original rule converges.
legacy_uniform_grazing_reference_remains_explicitly_unresolved|A failed measuring method must not be silently promoted after a new method succeeds.|Keep the original grazing reference rejection, its five-level limit and full evaluation count as regression evidence.
refined_reference_agrees_with_converged_original_and_exact_value|Changing sample placement must not change the lighting model.|Compare refined and original converged integrals and the independently derived classic closed form.
environment_reference_unresolved_diagnostics_are_not_results|An unfinished integration must never become a usable reference image.|Propagate the explicit convergence exception and retain its diagnostic history rather than returning a last estimate.
environment_preflight_rejects_unknown_material|A complete-material gate must not silently fall back to diffuse or another model.|Reject unsupported material names before integration.
saved_complete_environment_reference_has_two_valid_transitions|A saved reference must still satisfy its complete convergence and worker limits.|Verify its fixed report hash, both final qualifying transitions, sample count and actual monitored worker before displaying its radiance.
saved_public_environment_point_matches_reference_but_is_not_an_image|One correct surface point must not be promoted to a complete image or performance claim.|Recheck the saved actual public sampling result, reference tolerances and exact call count while retaining its one-point scope.
environment_progress_refuses_changed_report|Dashboard progress must not accept edited numerical evidence.|Change a saved report and require digest rejection before decoding it.
environment_progress_uses_accepted_producer_metadata|Correct measurements still need valid dashboard metadata to reach the viewer.|Validate the producer metadata against the dashboard protocol and verify it identifies the exact tested wheel; individual report hashes remain in the evidence panel.
'''

_CONST_ENV_GUIDES = '''
split_azimuth_weights_cover_half_hemisphere_and_polynomials|Concentrating angular samples must preserve the amount of light being integrated.|Check positive weights and exact polynomial moments for both independent half-azimuth partitions.
split_complete_classic_matches_exact_environment|The refined angular rule should resolve the classic grazing-view failure without relaxing accuracy.|Compare full classic integrals with exact results at four viewing angles, including the saved head minimum.
split_reference_keeps_work_limit_and_failure_history|Improved sampling must retain a hard numerical work ceiling.|Permit exactly one paired level, then require an unresolved result carrying its sample count and history.
split_reference_preserves_cooperative_timeout|A reference must stop when its time allocation expires.|Use an already effectively exhausted deadline and require an explicit time-limit failure.
split_reference_rejects_invalid_view|Invalid viewing directions must not produce believable light values.|Reject zero, back-facing, out-of-range and nonfinite viewing cosines.
square_radial_mapping_preserves_measure_and_moments|Extra samples near the horizon must not accidentally add or remove energy.|Test the squared radial coordinate with its exact Jacobian using positive weights and polynomial hemisphere moments.
square_radial_complete_classic_matches_known_answer|Changing the radial coordinate must preserve the full reflection integral.|Compare the restricted square-coordinate rule with exact classic answers at grazing, ordinary and normal views.
separated_matte_uses_residual_budget_not_plain_lambert|Separating an exact integral must retain the material's energy allocation.|Compare the constant-environment matte identity with independent scalar loss/mean integration and reject plain-albedo substitution; the glossy term remains separately required.
'''

_DIRECT_IMAGE_GUIDES = '''
image_reference_source_pin_matches_saved_corpus_revision|An image comparison must use the intended independent reference, not another module with a plausible name.|Check the source digest against the immutable corpus revision and the worker pin before accepting measurements.
prepared_scalar_curve_matches_original_without_reconstruction|Repeated pixels should not rebuild the full lighting table in the reference evaluator.|Prepare one fixed-roughness curve, block reconstruction, compare original scalar losses and mean, and reject a different roughness.
prepared_real_curve_retains_log_segments_and_independent_means|Reference preparation must preserve the accepted table's actual interpolation, not just a toy curve.|Compare horizon and logarithmic-angle queries plus adaptive mean against the original scalar evaluator on the full payload.
direct_tiles_compare_independent_pixels_and_keep_background|The public complete material should match independently computed pixels without retaining a large graph.|Observe eight-point forward tiles and a short final tile, compare scalar reference pixels and require exact black background.
image_tool_rejects_training_graphs_before_public_calls|A preview must not silently discard gradients supplied by a fitting application.|Send gradient-bearing normal, view or albedo buffers and require rejection before any public material call.
image_tool_rejects_invalid_buffers_before_reference_work|Bad geometry, masks or colour fields must not produce misleading image evidence.|Reject empty masks, wrong shapes or precision, nonfinite and zero directions, and out-of-range colour.
image_tool_honours_expired_deadline|A finite image job should stop when its allotted time is gone.|Pass an expired cooperative deadline and require an explicit timeout before reference work.
image_comparator_rejects_wrong_channels_light_and_background|The comparison must detect visibly wrong outputs, not merely report finite numbers.|Swap RGB, double illumination and change the background; each must fail the unchanged linear-image comparator.
explicit_public_tiles_preserve_shared_parameter_gradients|Small fitting tiles should retain gradients to shared lights and material parameters.|Preserve eight-point backward memory rejection, then compare four points together versus two-point tiles for every input derivative.
image_worker_refuses_existing_payload_before_launch|Repeating an experiment must not overwrite earlier evidence.|Place a prior image payload at the target and require refusal before any worker starts.
saved_image_reader_rejects_changed_report_before_loading_arrays|A changed measurement report must not silently select different evidence.|Supply a mismatched digest and require refusal before array loading.
saved_full_head_image_rechecks_hashes_and_linear_errors|A green full-head image result must remain reproducible from the actual saved pixels.|Verify report identities, worker limits, array hashes and immutable mask, then recompute every linear metric without rerunning shading.
saved_image_rejects_tampering_and_failed_supervision|Changed pixels, paths, masks or failed workers must not appear as passing image evidence.|Modify copies of the saved evidence and require hash, descriptor, supervision or recomputed-metric rejection.
'''

_LIGHT_LAYOUT_GUIDES = '''
public_lights_shared_strided_batches_keep_values_and_gradients|Multiple scenes must not mix illumination or lose shared fitting parameters.|Compare noncontiguous batches with separate scene calls in both precisions, including derivatives and repeatability.
public_light_empty_layouts_preserve_connected_gradients|Empty point or light selections should remain differentiable and predictable.|Check separate empty axes and connected zero gradients to all relevant floating inputs.
public_lights_reject_invalid_layout_domain_and_arithmetic|Malformed lights should produce explicit errors rather than silent physical changes.|Reject wrong shape, batch, precision and device, negative weights, nonfinite data and nonrepresentable inverse-square arithmetic.
public_lights_normalize_finite_extreme_nonzero_vectors|Valid directions should not become invalid merely because their scale is large or small.|Compare SH irradiance and normalized directional lights at very small and large finite vector scales without an epsilon floor.
public_light_demo_calls_all_three_installed_operations|The demonstration must use the lighting functions available to applications.|Observe all three public calls, check a known inverse-square value, bounded sphere image and retained negative environment ringing.
'''

_COMPLETE_GUIDES = '''
principled_oblique_value_matches_independent_scalar_formula|Check the whole reflection model away from straight-on illumination.|Compare the roughness-one BRDF with a separate scalar equation and a synthetic linear loss curve, never public lobe calls.
principled_reciprocity_preserves_values_and_input_gradients|Swapping viewing and lighting directions should preserve reciprocal reflection.|Compare full values and mapped derivatives at three roughness values with unchanged material fields.
complete_material_all_input_gradcheck|Fitting must reach every continuous material and direction input.|Finite-difference check classic and principled normals, view, light, colour and specular parameters together at smooth interior points.
principled_batched_strided_fields_match_separate_calls_and_gradients|Multiple scenes must not mix colours or lose shared gradients.|Compare a noncontiguous batched RGB field against independent scene calls, including derivatives and input immutability.
principled_inactive_pairs_have_zero_all_input_gradients|Unlit branches must not generate invalid half-vectors or division results.|Check view horizon, light horizon and opposite directions for exact zero values and all-input first derivatives.
principled_two_sided_equals_explicit_normal_flip|Two-sided shading should perform only its declared orientation change.|Compare a back-facing two-sided call with an explicitly flipped normal and otherwise identical inputs.
resource_queries_do_not_read_files_or_parse_json|Repeated shading should not silently read or decode stored data.|Load first, intercept Python file and JSON entry points, then query values and gradients from the resource.
resource_query_enforces_its_reported_workspace_requirement|A supplied scratch-memory limit must have an observable effect.|Reject one byte below the resource estimate and accept the estimate for value-only and backward workloads; not an allocator-peak proof.
resource_empty_strided_queries_keep_shapes_and_gradients|Empty and noncontiguous query layouts should retain predictable derivatives.|Compare a strided vector with scalar calls, then require an empty result with connected zero gradients.
'''


def _add(path, limits, rows):
    for row in rows.strip().splitlines():
        name, purpose, method = row.split('|')
        key = 'tests/' + path + '.py::test_' + name
        if key in GUIDES: raise ValueError('duplicate test guide: ' + key)
        GUIDES[key] = dict(purpose=purpose, method=method, limits=limits)


_add('python/test_render_backward_defaults', 'Hardware-free adapter regression, not physical CUDA or renderer qualification.', '''
raster_backward_missing_cotangent_device|Do unused raster gradients stay on the same device as the geometry?|Provide one of three gradient outputs while selecting an unrelated meta default; inspect all native-call cotangents for input device, dtype, shape and exact zero values.
raster_backward_all_missing_does_not_call_native|Does a backward call with no gradient work avoid the native kernel?|Replace the native entry with a fail-on-call sentinel and require ten absent input gradients.
''')
_add('python/test_m1_reporting', 'Dashboard routing and explanation checks only, never numerical operation acceptance.', '''
m1_page_assignment_is_explicit|Can I find each new M1 test family under an understandable page?|Resolve representative test identifiers to four fixed page IDs and check their test category, creation date and description.
m1_raster_regressions_have_authored_explanations|Can I understand what the new raster regression tests establish?|Require purpose, method and limitations for both device-ownership regressions.
''')


_add('python/test_environment_loss_demo', 'Finite sparse preflight visualization only. Swatches represent two selected points, not a full rendered image; normalized absolute gradient maps are display aids.', '''
environment_swatches_label_two_pixels_without_inventing_an_image|What do these two coloured rows represent?|Check exact target/before/after placement for two measured surface pixels and separately normalize the source derivative display.
environment_demo_publication_stays_finite_and_preserves_live_reporter|Does the four-profile preflight stay readable and bounded?|Require eighteen labelled artifacts, eight finite images and a clean finish without intercepting the active pytest reporter.
''')

_add('python/test_environment_loss', 'Two selected surface pixels per material/environment pair, CPU float64 and a frozen colour texture. Angular-adjoint preflight only, not full-image, light/geometry gradients or M1 acceptance.', '''
angular_loss_cotangent_is_not_a_sum_of_chunk_losses|Why must we add all the light before differentiating an image error?|Compare the complete squared-error derivative with a two-pass sum of sensitivities, and explicitly reject summing errors of partial light contributions.
environment_colour_slope_matches_classic_closed_form|Do surface-colour changes have the expected effect under world lighting?|Check the independent classic slope against exact hemisphere moments under uniform and coloured affine radiance.
streamed_native_texture_gradient_matches_small_complete_graph|Does releasing each angular graph preserve the gradient?|For both materials, compare a four-chunk recomputed adjoint through the native texture sampler with a small complete autograd graph, checking radiance, every texel derivative and work counts.
environment_preflight_rejects_invalid_work_before_shader|Can invalid profiles or excessive sampling reach the shader?|Forbid shader calls while testing unknown lighting, mismatched point counts, expired deadlines and an excessive angular order.
environment_schedule_and_existing_root_fail_closed|Are the experiment profiles and saved outputs protected?|Require the exact four material/light pairs and refuse an existing output directory.
environment_failed_preflight_preserves_stop_and_prevents_next_profile|Will a failed environment trial automatically start more work?|Simulate one stopped worker and require a persisted stop with no second profile launched.
saved_environment_loss_requires_explicit_verified_evidence|Can these saved gradients be trusted after the process exits?|Reconstruct the independent reference and adjoint, check both offsets and fitting steps, then reject an incorrect hash and ten rehashed forgeries of pixels, limits, identities, work, targets, derivatives, losses, offsets, updates and finite differences.
''')

_add('python/test_image_loss_demo', 'Finite presentation of separately replayed image-loss evidence. sRGB and normalized absolute gradient maps are display transforms, never numerical acceptance.', '''
fitting_panels_preserve_target_before_after_and_background|Can I tell the target from the two measured fitting images?|Check exact left-to-right placement, black background and the explicitly normalized source-gradient map.
fitting_publisher_has_finite_explained_artifacts_without_intercepting_live_tests|Will this demonstration stay small and keep live test reporting intact?|Publish through local fakes, require ten labelled artifacts and a clean finish, and preserve the actual pytest reporter method.
''')

_add('python/test_image_loss_gradients', 'Fixed CPU float64 direct light, one 32x32 texture level, frozen geometry and UVs. Source-colour derivatives only; no environment, geometry, visibility, CUDA or complete M1 claim.', '''
bilinear_stencil_repeat_half_texel_and_partition|Which texture pixels contribute to a rendered point?|Check the independent four-texel weights at texel centres, wrap seams and repeated coordinates, and reject malformed UVs.
reference_source_adjoint_and_covered_denominator_known_answer|Does image error flow back to the correct source texel?|Compare a one-pixel squared-error example with its hand-calculated derivative, including the RGB denominator and exact zero at untouched texels.
reference_adjoint_matches_both_finite_difference_scales|Is the independent gradient itself trustworthy?|Perturb three fixed texture patterns at two small scales and compare loss differences with the scalar bilinear transpose, without using automatic differentiation.
streamed_texture_loss_matches_reference_and_second_step|Can an image loss actually improve the source texture?|Use the installed native bilinear sampler and complete material, compare RGB, loss and source gradients to an independent calculation, then build a fresh graph and require a second step to reduce loss.
invalid_fixture_fails_before_public_shader|Do malformed fitting inputs fail before shading work starts?|Reject empty, nonfinite, mismatched, zero-direction, unknown-model and expired-deadline fixtures while forbidding the public classic shader.
exact_schedule_and_create_only_execution|Will this experiment stay bounded and preserve earlier evidence?|Require two 64-point preflights followed by two complete 6536-point images, and refuse an existing output directory.
failed_preflight_never_launches_full_image|Does a failed small trial stop the larger experiment?|Simulate a stopped first worker; require exactly one launch and a persisted stopped checkpoint, never a full-image retry.
pass_validation_rejects_wrong_measurements|Can a plausible-looking but wrong gradient be accepted?|Corrupt rendered colour, source derivative, scalar loss or an untouched texel and require validation to reject it.
saved_image_loss_evidence_rejects_corruption|Do saved fitting results still belong to the exact experiment?|Require an explicitly selected four-worker cohort, replay its identities and numerical comparisons, and reject a forged report digest.
''')


_add('python/test_albedo_images', 'Bounded CPU float64 validation tooling for the twelve fixed textured image profiles. New pixels require independent references; a partial checkpoint is not full-image, gradient, arbitrary HDR or M1 acceptance.', '''
new_pixels_need_reference_not_previous_public_outputs|Can a previously untested surface point be shaded honestly?|Evaluate nine points from reference-only inputs, preserve the measured per-profile route and compare both angular estimates to a separately computed public baseline.
new_pixels_reject_mismatched_reference_before_public_call|Does the reference actually belong to this material, light and colour?|Forge each identity and forbid public calls; reject the mismatch before running the material and reject an expired deadline.
reference_only_result_rejects_forged_counts_routes_and_values|Can wrong work or a bad colour appear to pass?|Change call counts, direction counts, pair limits, route, timing and RGB values; require independent replay to reject each forgery.
full_image_schedule_excludes_all_selected_tiles_and_is_create_only|Will continuation skip completed work without losing any remaining pixels?|Partition exactly588 unique remaining tiles into98 batches, exclude all36 selected checkpoints, and reject overwriting output or launching without time reserve.
new_tile_inputs_match_raster_order_and_saved_albedo|Does a new tile use the actual saved image colours and geometry?|Read the immutable corpus, match the second covered-pixel tile to raster-order IDs and all three saved colour fields, and reject invalid indices.
frozen_sampling_observations_match_reference_only_sampler|Did removing the old-output requirement change the numerical answer?|Strictly replay the selected parent evidence, discard only the old public inputs, freshly shade all72 saved observations and compare at the stricter regression tolerance.
saved_new_tile_batch_replays_and_rejects_forgery|Are newly completed tiles backed by trustworthy evidence?|Replay six exact new tiles and reject altered pixel order, colour, reference, counts, route, source identity, worker memory and failure state.
partial_progress_panels_show_only_computed_pixels|Can the preview accidentally present missing work as a black rendered pixel?|Draw a small known checkpoint, verify untouched pixels stay gray, preserve input evidence and reject duplicate ownership.
''')

_add('python/test_albedo_image_assembly', 'Complete-image assembly of the fixed CPU shader profile, using strictly replayed selected/new checkpoints. Synthetic ownership tests are not physical image measurements. Passing one profile does not qualify the other eleven, independent rasterization, image gradients or M1.', '''
complete_assembly_preserves_all_pixels_and_exact_background|Does a complete image place every measured colour in the correct pixel?|Assemble all52 tiles over a known mask, compare both RGB images, require exact zero background and confirm input evidence is unchanged.
complete_assembly_rejects_incomplete_or_wrong_evidence|Can a partial or corrupted image accidentally look accepted?|Remove, duplicate or reorder tiles; change profile, pixel, status, tail length or numerical values; every variant must be rejected.
assembly_requires_explicit_ordered_batch_hashes_and_preserves_output|Can assembly silently choose different checkpoints or overwrite a prior image?|Require explicit ordered identities, reject an existing output directory, and reject a changed report before parsing or reconstruction.
saved_complete_textured_image_reconstructs_and_rejects_changed_bytes|Can the saved full image be reproduced from its original measurements?|Strictly replay the actual52 tiles, recompute image metrics and byte-identical arrays, then flip a saved image byte and require rejection.
complete_image_panels_keep_background_and_separate_reference|Does the completed-image view preserve the accepted background and both answers?|Render the actual and reference arrays side by side, check zero background, and preserve their left/right identity.
matrix_dashboard_shows_each_profile_once_within_artifact_limit|Will all twelve completed profiles fit on one dashboard page without duplicate images?|Simulate zero, two and twelve accepted profiles; require exactly twelve image panels, keep complete evidence records, omit duplicate partial views and stay below the unchanged32-artifact limit.
''')

_add('python/test_albedo_sampling', 'CPU float64 development sampler on fixed saved profiles. The 72-pixel timing screen reuses observed regression data, not fresh physical holdouts; it does not qualify full images, gradients, GPU or general speed guarantees.', '''
batched_public_sampling_matches_independent_and_saved_outputs|Batching must preserve the actual shaded colours, not just produce plausible images.|Compare both angular estimates against the independent reference and original public output for each material and lighting type; count all directions and enforce pair/point caps.
sampling_preserves_partial_groups_and_rejects_bad_input_before_calls|The eight-point limit and an incomplete final group must work without hiding invalid inputs.|Observe real public calls for nine points, require groups of eight and one under 1024 pairs, and reject invalid colour or an expired deadline before any additional shader call.
sampling_groups_keep_quadrature_orders_separate|Points needing different integration resolutions cannot silently share a lower-resolution grid.|Require deterministic grouping by accepted order, preserve original point indices and reject unsupported widths.
sampling_validation_rejects_forged_work_and_pixels|A saved speed result must not hide missing work or changed colours.|Corrupt direction/call counts, pair bounds and RGB values and require replay rejection.
sampling_screen_schedule_and_history_guard|Declare the experiment before timing it and preserve previous runs.|Require six pinned parent identities, 72 selected pixels, the full eight-pass direction ceiling and create-only output.
saved_sampling_screen_replays_all_profiles|Any timing decision needs exact source, parent, numerical and worker evidence.|Replay all four profiles against the immutable tiles; verify alternating method order, measured repetitions, medians, parity and process limits.
saved_sampling_replay_rejects_changed_parent_timing_and_supervision|A new hash alone cannot legitimize altered benchmark evidence.|Change parent identities, selected rows, execution order, durations, preference flags and worker supervision, then require rejection without changing saved files.
''')

_add('python/test_albedo_tiles', 'Selected CPU float64 image tiles at fixed material parameters. Each profile checks264 of6536 surface pixels, not complete images, independent rasterization, image gradients, GPU or M1 acceptance.', '''
tile_schedule_has_exact_unique_profile_ownership|Every selected pixel group must belong to exactly one declared image profile and batch.|Require36 unique material/environment/colour/tile combinations and six ordered batches; reject invalid batch indices.
tile_launch_guard_and_destination_preserve_history|A batch must preserve earlier results and reserve enough time for a complete worker.|Reject an existing output directory and a launch with less than180 seconds remaining.
tile_reader_checks_hash_before_parsing|Changed report bytes cannot be treated as measured evidence.|Supply bytes with the wrong digest and require rejection before interpreting any tile fields.
selected_tiles_use_exact_saved_fields_and_short_tail|The first, middle and final tiles must use their actual saved colours and pixel positions.|Check128,128,8 unique pixels per colour field, including the short final tile, against immutable image buffers.
tile_input_rejects_unknown_field_and_index|Only the prospectively selected image slices belong to this experiment.|Reject constant-colour substitution, a nonselected tile and a boolean masquerading as an index.
aggregate_requires_all_six_batches_without_duplicates|Do not hide missing profiles behind a completed-subset label.|Reject an empty aggregate and duplicate batch ownership.
completed_tiles_replay_with_corruption_controls|Acceptance requires every declared pixel and genuine bounded worker evidence.|Replay all36 tiles and3168 comparisons, then reject missing or reordered pixels, altered colour or environment, excessive memory and forged work counts.
partial_image_panels_keep_uncomputed_pixels_gray|An unmeasured pixel must never look like a verified black background.|Check the gray uncomputed region, separate reference/public panels, and duplicate pixel refusal using a tiny deterministic display fixture.
''')

_add('python/test_albedo_environment', 'CPU float64 saved surface-point preflight at fixed material parameters. Not complete textured images, image-loss derivatives, general HDR loading, GPU or M1 acceptance.', '''
albedo_preflight_schedule_and_work_caps_are_fixed|Know exactly which material, light and colour combinations will be tested before execution.|Require four ordered workers, 96 comparisons, the declared direction ceiling and a full watchdog reserve before launch.
albedo_validation_is_strict|A surface colour must be a valid reflectance, not a misleading invalid number.|Accept finite RGB in zero to one; reject bad lengths, booleans, nonfinite and out-of-range channels.
generalized_reference_preserves_the_qualified_constant_colour|Adding changing colours must not alter the previous fixed-colour reference.|Compare the generalized independent reference against the frozen accepted reference for both materials in float64.
uniform_classic_albedo_has_independent_closed_form|Anchor the numerical measuring method to an exact mathematical answer.|Integrate black and nonuniform RGB colours at normal view and compare with diffuse reflectance plus the exact classic glossy integral.
albedo_reference_honours_work_and_deadline_limits|New colour profiles must retain bounded execution.|Force a one-level sample allowance and an expired deadline, requiring explicit failure rather than an unqualified estimate.
reference_and_candidate_refuse_changed_albedo_identity|A comparison cannot silently use different input colours or fabricated convergence.|Change the candidate colour and the saved convergence streak; require rejection of each.
albedo_cohort_refuses_overwrite|Keep earlier measurements recoverable instead of replacing them.|Refuse an existing cohort directory before launching any workers.
public_nonconstant_colour_matches_reference_and_detects_corruption|The real public material and light integrator must agree with the independent colour-aware reference.|Compare both materials under both environments, then corrupt colour, environment, work counts, error arrays and estimates; require rejection.
saved_albedo_input_selection_is_exact_and_preserves_fields|Use the immutable vertex, gradient and checker fields at the declared points, not substitute colours.|Check all 24 ordered field/rank selections against the source corpus values and unique selected pixel IDs.
saved_albedo_preflight_replays_exact_fields_and_all_comparisons|Point acceptance needs the full declared experiment and reproducible evidence.|Replay all four workers and 96 comparisons against corpus, source, wheel, colour, convergence, work and memory identities.
albedo_swatches_preserve_input_reference_and_public_order|The demonstration must not swap input colour and measured outputs.|Check exact swatch dimensions, colour ordering and separation for all three saved fields.
saved_albedo_replay_rejects_changed_field_order_and_worker_evidence|Rehashed or rearranged evidence must not quietly become accepted measurements.|Reorder selected points, alter colours or field labels, and inject worker failure, memory overflow or PID mismatch; require rejection without modifying saved artifacts.
''')

_add('python/test_environment_full_images', 'Fixed constant-environment CPU image profile only. Shared geometry is not independently validated here; nonconstant lighting, image gradients, CUDA and M1 acceptance remain separate.', '''
fixed_profile_reference_samples_match_frozen_scalar|The fast measuring method must compute the same physical formula as the independent scalar evaluator.|Compare classic and roughness-one principled glossy samples at grazing, ordinary and normal views without calling the public shader in the reference.
fast_reference_matches_scalar_convergence_history|Faster integration must not hide a reference that has not converged.|Require the same evaluation count and two-transition history as the scalar method at the head fixture's most difficult grazing angle.
reference_work_and_time_fail_closed|Reference work must stop within its declared limits.|Force a one-level work budget and an expired deadline; require explicit failure rather than the last estimate.
reference_rejects_invalid_pixels|Invalid viewing directions must never become plausible reference colours.|Reject nonpositive, out-of-range and nonfinite viewing cosines.
tile_layout_covers_every_pixel_once_and_rejects_bad_indices|Splitting an image must neither skip nor duplicate visible pixels.|Check all 52 exact tile ranges cover 6536 pixels once, including the short final tile, and reject invalid indices.
public_pixel_uses_two_estimates_and_rejects_bad_reference|Correct-looking averages must not hide a failed integration estimate.|Compare both actual public-lighting estimates with the converged reference and reject a deliberately wrong reference colour.
local_frame_matches_world_frame_public_integral|Local coordinates must preserve isotropic lighting rather than rotate the illumination accidentally.|Integrate the public classic material in original and rotated frames with identical quadrature weights.
assembly_refuses_missing_duplicate_or_failed_tiles|Partial image progress must not be promoted to complete acceptance.|Refuse missing tile sets, duplicate ownership and failed evidence before image assembly.
''')

_add('python/test_affine_images', 'Fixed affine CPU float64 image profile only; original checkpoints remain immutable. No arbitrary HDR maps, changing roughness or M1 completion claim.', '''
remaining_schedule_covers_only_unmeasured_tiles|Complete the image without wasting or overwriting finished work.|Require 98 new unique tiles to complement the six qualified parent tiles, covering 12544 additional pixels.
batches_have_finite_unique_ownership_and_watchdog_reserve|Bound each batch while allowing the project to keep moving.|Require seventeen ordered batches of at most six tiles, the fixed work ceiling and enough time left for one complete worker watchdog.
parent_identity_is_required_before_reuse|Only the exact qualified parent can supply reused results.|Reject changed parent bytes before reuse.
full_run_refuses_existing_output|A rerun must never erase evidence of an earlier attempt.|Reject an existing output directory before starting a worker.
image_assembly_rejects_missing_or_misordered_tiles|Partial or shuffled coverage must never become a complete image.|Refuse missing records and an incorrect first tile.
batch_layout_rejects_missing_duplicate_or_reordered_descriptors|Checkpoint metadata must name the exact work completed.|Validate a correct prospective batch, then reject duplicate and missing descriptors.
saved_affine_images_replay_complete_pixel_coverage|Full acceptance needs every original and new pixel, not a passing subset.|Rebuild both complete images from hash-bound validated tiles and compare saved arrays, metrics and execution accounting.
assembly_reconstructs_every_pixel_and_rejects_invalid_estimates|Check the final image assembler independently of expensive shading.|Assemble a known constant image, require exact background, and reject NaN estimates and wrong pixel ownership.
live_checkpoint_demo_marks_only_verified_progress|Keep a progress display honest while calculations continue.|Read a finite hash-bound checkpoint snapshot, validate completed batches, and require the known-pixel mask to match only verified coverage.
''')

_add('python/test_cuda_repetition', 'Twenty fixed same-device repeats, not a sustained-memory guarantee, public GPU admission or cross-device bit-identity.', '''
cuda_repeat_harness_has_finite_fixed_limits|Declare the CUDA test size and resource ceilings before running it.|Require twenty repetitions, 2 GiB whole-process host memory, 64 MiB allocated GPU memory, a 180-second watchdog and at most 64 KiB post-warm live allocation growth.
cuda_material_repetition_has_fresh_graphs_and_bounded_live_allocations|Repeated fitting calls must not retain old graphs or mutate fixed material resources.|Rebuild and differentiate the maximum 1024-pair material workload twenty times in deterministic mode, compare exact outputs/gradients, verify unchanged constants and measure live and peak GPU allocations separately from reserved memory.
''')

_add('python/test_shading_m1_gradients', 'Small installed-wheel smooth-interior fixtures. CPU uses the accepted budget. Classic/SH CUDA parity does not qualify the principled CUDA resource or a full fitting engine.', '''
complete_shading_mip_chain_all_input_gradcheck|A fit must reach the original texture and every continuous shading input.|Finite-difference check two-level mip sampling through complete classic or principled shading and lighting, including texture, UV, normal, view, directions, material parameters and incoming light.
two_shading_steps_rebuild_mip_graph_and_reduce_loss|A second optimization step must use a fresh graph rather than stale cached shading.|Update source texels using the first gradient, rebuild the pyramid and shading, and require lower loss plus a valid new gradient.
classic_and_sh_cuda_values_and_gradients_match_cpu|Already tensor-based shading should preserve its meaning on the GPU.|Compare classic BRDF and all SH coefficients on CPU versus CUDA, including input derivatives in float32 and float64.
cuda_mip_texture_shading_chain_matches_cpu_all_partials|GPU fitting must reach the original texture through the native mip-map and material chain.|Compare two-level CUDA texture sampling, complete material and light integration against CPU outputs and every source-texture, UV, geometry, light and material partial. Principled CUDA remains explicitly experimental.
''')

_add('python/test_principled_cuda_qualification', 'Explicit experimental CUDA float64 only. Casting a CPU resource never grants CUDA physical admission; full physical, memory and artifact qualification remain separate.', '''
cuda_transfer_is_explicit_and_cannot_inherit_cpu_admission|Moving constants to a GPU must not fabricate acceptance evidence.|Reject default unqualified transfer; explicit research transfer preserves payload, moves every constant and has no accepted CUDA profile. Returning exact data to CPU recovers only its existing CPU acceptance.
cuda_budget_queries_and_partials_match_cpu|The GPU curve must preserve values and fitting derivatives at ordinary and grazing points.|Compare directional-plus-mean loss and cosine, roughness and Fresnel partials at three fixed points against CPU float64.
cuda_budget_rejects_wrong_query_device_and_preserves_input|A GPU resource must not silently move CPU inputs.|Reject CPU query tensors against CUDA constants and verify the caller tensor remains unchanged.
cuda_complete_material_matches_cpu_values_and_all_partials|Test the full reciprocal matte and GGX composition, not only curve queries.|Require separate experimental material opt-in and compare complete outputs and all input partials with the accepted CPU path.
cuda_supervision_changes_only_host_cap_and_restores_cpu_policy|Keep the approved CUDA memory allowance separate from CPU acceptance limits.|Inject a worker failure and require only the CUDA host cap to become 2 GiB, with the 180-second watchdog unchanged and the original CPU profile restored afterward.
''')

_add('python/test_principled_cuda_physical', 'Supervised CUDA-f64 resource screen on the previously guarded cohort; not unseen holdouts, public catalog admission, GPU rendered-image acceptance or a sustained-memory guarantee.', '''
cuda_directional_values_and_gradients|Check GPU directional losses against independently calculated reflection energy and derivatives.|Run the fixed eighty-point public profile with actual CUDA queries, independent CPU references and unchanged physical/refinement thresholds.
cuda_mean_values_and_gradients|The GPU average must remain the integral of its own loss curve and preserve its derivatives.|Compare sixteen roughness/Fresnel pairs against independent curve integration and physical hemisphere references.
cuda_furnace_energy_balance|The GPU budget must not invent or lose energy beyond the declared physical tolerance.|Check 192 black, gray and white furnace cases against independent specular and scalar curve integration, preserving coarse/fine refinement checks.
''')

_add('python/test_affine_tiles', 'Six selected CPU float64 image tiles only. Reusing material setup must not reuse pixel answers. Not full images, arbitrary HDR maps, gradients, GPU or M1 acceptance.', '''
tile_schedule_is_fixed_partial_and_includes_short_final_tile|Measure preselected beginning, middle and final regions without cherry-picking successful pixels.|Require exactly six tile identities and 528 comparisons, including the short final tile and the prospective sample ceiling.
prepared_reference_preserves_fresh_point_history|Sharing fixed setup must not change the independent mathematical answer.|Compare the complete reference result and every refinement with a fresh original calculation for both materials.
prepared_setup_runs_once_and_does_not_share_returned_history|Avoid repeated material setup without letting one pixel corrupt another.|Count exactly one moment computation; mutate one returned history and require the next result to remain unchanged.
prepared_reference_owns_copy_of_caller_curve|A caller changing its curve must not silently change a running qualification.|Mutate the caller curve after preparation and require identical reference output.
prepared_reference_still_checks_each_pixel_and_deadline|Setup reuse must retain per-pixel geometry and execution limits.|Reject behind-surface views and expired deadlines; require different views to produce their own radiance.
prepared_reference_matches_all_saved_point_histories|The prepared path must agree with all previously qualified points, not just a convenient example.|Replay the immutable sixteen-point cohort and require exact complete reference histories under worker-owned setup.
tile_replay_rejects_wrong_identity_before_parsing|Unverified report bytes must not become accepted evidence.|Require a SHA-256 mismatch to fail before report parsing.
tile_run_refuses_overwrite_and_reserves_watchdog|A measurement must preserve existing evidence and leave time to stop safely.|Reject an existing output directory and a launch without a full watchdog interval remaining.
partial_tile_cohort_cannot_claim_full_images|A changed or partial experiment must not become image-wide acceptance.|Reject an altered cohort identity; the successful saved replay separately forbids an images claim.
saved_tile_qualification_replays_all_selected_pixels|Every selected pixel must have qualified saved evidence.|Replay six exact tile identities and all 528 geometry, convergence, public-output and resource checks.
pixel_replay_rejects_corrupted_ownership_and_numerics|A passing label must not hide wrong pixels, fabricated work or invalid numerical results.|Tamper with ownership, geometry, estimate shape, sample counts, error values, NaN and moment counts and require rejection.
tile_demo_marks_uncomputed_pixels_and_rejects_duplicate_ownership|A partial-image diagram must make missing evidence obvious.|Require green only for measured pixels, gray elsewhere, and reject duplicate ownership.
''')

_add('python/test_affine_environment', 'Fixed CPU float64 affine world-lighting preflight: selected points, not full images, arbitrary HDR-map loading, image gradients or M1 acceptance.', '''
full_sphere_quadrature_preserves_hemisphere_moments|Directional light needs the whole visible hemisphere, with the correct amount of solid angle.|Check unit samples, positive weights, total 2pi measure, first moments and squared coordinate moments for both full-azimuth partitions.
affine_radiance_is_positive_and_matches_independent_sh_ordering|Coloured illumination must retain the intended world axes and colour channels.|Compare the affine world field to independently evaluated SH coefficients, including y,z,x ordering and strictly positive radiance.
affine_field_rejects_unqualified_coefficients|Invalid or potentially negative-light profiles must fail before sampling.|Reject oversized, nonfinite, malformed and nonrepresentable coefficient triples.
view_frame_is_orthonormal_and_retains_world_orientation|A local integration frame must not silently rotate the scene lighting.|Check a right-handed orthonormal basis and the local view components for normal and grazing configurations.
reference_rejects_invalid_or_backfacing_geometry|Undefined frames must not produce plausible lighting measurements.|Reject zero, nonfinite, tangent and behind-surface views.
affine_lambert_integral_matches_public_sh_convolution|Use a simple material with an exact directional-lighting answer to check the integration measure.|Compare numerical Lambert integration, the analytic affine formula and the public SH diffuse convolution.
weighted_loss_moments_match_known_polynomial_and_keep_history|The principled matte term needs the residual loss curve, not a plain diffuse approximation.|Integrate a polynomial control with exact moments and require both successive refinement checks to pass.
weighted_loss_moments_reject_disagreeing_mean_and_exhausted_work|An unreliable matte reference must stop the comparison rather than hide its error.|Require rejection of an inconsistent adaptive mean and an exhausted moment-sample budget.
affine_reference_keeps_work_and_time_guards|A more general lighting fixture must retain bounded numerical work.|Exhaust the direction budget after one level and verify its history; reject an expired deadline.
affine_reference_matches_independent_scalar_at_normal_view|The fast reference must implement the same complete material as the separate scalar evaluator.|Compare against independent full-azimuth scalar integration; a polynomial residual is a numerical control, not resource admission.
affine_public_complete_material_matches_reference_at_grazing_view|Difficult edge-on views must work through the actual public material and light integrator.|Compare both public full-hemisphere estimates to the independently converged reference and reject tampered reference values.
affine_reference_rotates_lighting_with_geometry_and_detects_fixed_world_light|Moving the object and lights together should preserve shading; moving only the object should change it.|Rotate geometry and colour-direction coefficients together, then intentionally leave illumination fixed as a negative control.
affine_cohort_refuses_existing_output_and_reserves_watchdog|Preflight must neither overwrite history nor start a worker outside its allotted time.|Refuse an existing destination and reject launches with less than a full watchdog interval remaining.
affine_replay_refuses_changed_report_before_loading|Saved point evidence must have the exact expected identity.|Alter a report and require hash rejection before parsing its result.
affine_reference_replay_rejects_tampered_histories|A green status cannot replace evidence of convergence and bounded work.|Tamper with moment means, counts, shapes, values and angular histories; require rejection.
world_lighting_cannot_use_doubled_half_sphere_shortcut|Uniform-light symmetry no longer applies when opposite directions have different colours.|Show that doubling only one azimuth half changes the complete classic response measurably.
affine_cohort_replay_rejects_changed_summary|The final summary must refer to the exact declared experiment.|Reject a changed cohort hash before opening any point reports.
saved_affine_cohort_replays_all_predeclared_points|Measured acceptance needs every predeclared point, not a successful subset.|Replay sixteen unique model/rank reports against immutable geometry, source/wheel identities, worker limits and both numerical estimates.
affine_demo_map_is_finite_and_has_opposite_direction_colour_balance|The illumination diagram must faithfully represent the declared affine field.|Check finite positive image values and the exact opposite-direction identity L(w)+L(-w)=2.
''')

_add('python/test_environment_continuation', 'Only the approved constant-environment continuation. Replayed checkpoints are not new physical holdouts, general environment maps or full M1 acceptance.', '''
continuation_schedule_contains_only_unfinished_tiles|Continue useful work without recomputing finished images.|Require exactly tile indices 32 through 51, 900 additional seconds and the declared additional evaluation ceiling.
continuation_reserves_watchdog_before_launch|A new worker must fit within the remaining authorized execution budget.|Require refusal once fewer than 180 seconds remain, while allowing an unexpired budget.
parent_digest_rejected_before_child_reads|A continuation must start from the exact previously reviewed checkpoint.|Change the parent report and require rejection before opening its tile files.
parent_layout_rejects_unqualified_prefix|Missing, duplicated or failed work must not become an accepted prefix.|Mutate status, tile count, ordering, paths, completed-image identity, stop reason and elapsed time; require explicit rejection.
continuation_refuses_existing_output_before_launch|Preserve both original and continued evidence instead of overwriting history.|Supply an existing destination and require refusal before any worker starts.
generator_source_identity_is_frozen|All image tiles must use the same qualified numerical implementation.|Check the six frozen source hashes and reject a changed source map.
saved_parent_prefix_validates_without_shading|Reusing evidence should verify it, not quietly rerender it.|Disable the candidate shader and replay the completed classic image and 32 principled tiles against immutable geometry and saved convergence histories.
continuation_replay_refuses_changed_summary|Dashboard replay must not accept an edited continuation result.|Require the caller's exact continuation digest before reading parent or image data.
saved_continuation_replays_both_complete_images|The final image needs every pixel checked across both execution segments.|Verify the original and additional budgets separately, reconstruct both 6536-pixel images from 104 unique qualified tiles and compare saved image bytes.
dashboard_projection_preserves_execution_segments|A dashboard must not disguise a stopped run as one uninterrupted success.|Combine only the displayed image evidence, retain original stopped status and separate budgets/timings, count the 20 new tiles and leave both source records unchanged.
''')

_add('python/test_environment_image_replay', 'Saved evidence verification, not fresh rendering or general environment acceptance. A missing complete cohort is an explicit skip, never a pass.', '''
replay_rejects_changed_cohort_before_reading_tiles|Edited reports must not quietly select different numerical evidence.|Require the caller-provided cohort hash before opening any child report or image array.
replay_recomputes_convergence_instead_of_trusting_flags|A success label must not conceal a broken reference integration.|Recompute thresholds, estimate differences, transition streaks and sample counts; reject injected nonfinite values or altered evidence.
replay_rejects_unknown_model_and_incomplete_cohort|A partial run must not be presented as complete full-image acceptance.|Require the exact known profile and completed cohort state before replaying the two-image acceptance set.
assembly_positive_and_pixel_tamper_controls|Every visible pixel needs exactly one owner, including the final short tile.|Assemble a complete synthetic evidence set, verify exact zero error and background, then duplicate the final pixel and require rejection.
saved_complete_environment_images_replay_all_pixels|Measured full-image results must be reproducible from their saved arrays and source identities.|Load an explicitly named and hash-pinned cohort, replay every reference history and public comparison, reconstruct both images and require all 6536 pixels per image to pass.
stopped_cohort_preserves_complete_images_without_promoting_partial_work|Stopping a cohort must preserve valid finished images without inventing an unfinished one.|Require full-cohort acceptance to reject the stopped run, then verify only its completed classic image from all saved pixels and immutable tile identities.
''')

_add('python/test_material_light_batches', 'CPU float64 material-only qualification. Standalone resource queries remain limited to 32 entries; full images, CUDA and sustained-use evidence are separate.', '''
exact_material_pair_capacity_is_accepted|A supported batch should work exactly at its stated boundary.|Evaluate all four forward/backward capacity corners and check finite output and graph ownership.
material_capacity_rejects_before_mask_or_curve|Oversized requests must fail before allocating expanded working state.|Intercept geometry construction and reject pair or point counts above the approved capacity before masks.
material_workspace_exact_byte_boundary|The caller memory limit must be enforced precisely.|Reject one byte below the declared estimate and accept the exact estimate for full-size forward and backward batches.
standalone_resource_query_limit_remains_32|A material optimization must not silently enlarge unrelated resource APIs.|Reject a 33-entry directional query and verify the unchanged resource metadata limit.
empty_material_keeps_zero_gradients|Empty work still needs predictable shapes and connected zero derivatives.|Exercise empty points and lights, including a no-light point array larger than the nonempty capacity.
inactive_large_batch_keeps_zero_gradients_and_counts_capacity|Invisible samples must not bypass capacity guards or produce invalid gradients.|Require exact zero output and derivatives for 1024 inactive pairs, then reject the 1025th pair.
batched_strided_large_material_matches_small_tiles_and_all_gradients|Batching must preserve every scene, point, channel and fitting derivative.|Compare a strided two-scene 1024-pair call with explicit eight-light tiles, accumulating reference gradients one tile at a time to bound memory.
environment_sampler_refuses_unqualified_chunk|The measuring tool must not silently request an unsupported batch.|Reject zero, boolean and oversized chunk values before integration.
environment_chunking_preserves_samples_and_weights|Changing call size must not change the lighting experiment.|Use identical quadrature samples and weights with small and large chunks and compare both RGB integrals.
large_cost_probe_reports_all_four_capacity_corners|Memory qualification must cover both point-heavy and light-heavy calls.|Check four finite supervised-workload definitions, twelve repetitions each and unchanged planning-byte ceilings.
no_grad_uses_forward_point_capacity|Trainable inputs should not incur backward limits when graph construction is explicitly disabled.|Evaluate eight points under no_grad and require forward-only output.
point_capacity_counts_across_batches|Each scene must count toward the total supported point limit.|Reject two five-point scenes before material evaluation rather than applying the limit per scene.
saved_large_batch_evidence_keeps_limits_and_accuracy|Recorded speed must retain its exact numerical work and safety evidence.|Check fixed report hashes, monitored worker, unchanged memory ceilings and the full reference comparison.
batch_demo_rejects_changed_report|Demonstrations must not promote edited measurement files.|Tamper with a saved report and require hash rejection before publication.
''')


_add('python/test_principled_direct_images', 'Bounded CPU float64 direct-image evidence with shared geometry. The forward-only tool is not a training renderer; physical holdouts, complete environment integration, CUDA/export and real-model M1 remain separate.', _DIRECT_IMAGE_GUIDES)
_add('python/test_environment_image_preflight', 'Independent reference controls, not public full-image, GPU or M1 acceptance. Failed rules remain explicit; no tolerance increase.', _ENV_IMAGE_GUIDES)
_add('python/test_principled_curve_reuse', 'CPU float64 only. Existing 32-entry and 64 MiB planning caps remain unchanged; sampled working set and saved tensors are different measurements, not a full-image, GPU or allocator-peak claim.', '''
call_local_losses_match_three_queries_and_all_partials|Reusing a calculation must preserve every colour and fitting derivative.|Compare shared curves with three separate resource queries, including repeated point ownership and horizon endpoints, at unchanged strict tolerances.
material_builds_one_curve_per_point_without_retaining_graph|Speedups must not accidentally reuse stale fitting state.|Observe one curve per active point, change roughness between two backward passes and require fresh results and graphs.
material_rejects_old_limit_before_any_curve|A smaller implementation must not silently raise the supported workload.|Intercept curve allocation and require the original eight-point backward workspace rejection first.
curve_reuse_preserves_two_node_resource|The smallest supported research table must keep its linear interpolation semantics.|Compare the shared path with separate public queries at horizon, ordinary and normal directions on a two-node angular resource.
cost_probe_stops_at_expired_deadline|Measurements must obey their time limit.|Supply an exhausted deadline and require the cost probe to stop before material evaluation.
cost_probe_reports_memory_categories_separately|Reported scratch estimates must not masquerade as measured process memory.|Check the three finite workloads, saved-autograd storage and twelve working-set samples per workload under unchanged bounds.
horizon_partial_matches_linear_segment_without_cancellation|Near the horizon, subtracting large derivative terms can lose the small answer.|Compare the derivative to the linear segment slope formed by subtracting endpoint values before division; no tolerance relaxation.
saved_reuse_measurements_retain_exact_work_and_limits|A speedup must refer to the same work and retain its safety evidence.|Verify fixed report hashes, actual monitored worker identity, unchanged call counts and independent-reference or memory bounds.
reuse_demo_refuses_changed_measurements|The demonstration must not display edited timing data as verified evidence.|Change a saved report and require digest rejection before accepting its contents.
''')
CHECK_GUIDES[('tests/python/test_environment_image_preflight.py::test_saved_complete_environment_reference_has_two_valid_transitions','Converged independent full-material reference')]=dict(
    purpose='Verify this saved surface sample has a trustworthy independent lighting reference.',
    method='Check its fixed report digest, monitored worker and sample limits, and both final convergence transitions.',
    limits='One independent reference point, not public full-image or GPU acceptance.')
_add('python/test_constant_environment_reference', 'Restricted constant-unit environment and isotropic materials only; not arbitrary SH transport. Same two-transition thresholds and finite sample/time ceilings.', _CONST_ENV_GUIDES)
_add('python/test_environment_candidate_point', 'Actual public material and light integration, forward-only at one CPU point. No private candidate substitution or full-image acceptance.', '''
public_environment_sampling_calls_full_material_and_integrator|An image-execution test must actually exercise the public shader, not just an analytical formula.|Observe every complete material and tagged light integration call, check ten-direction chunks and compare a known-answer integral.
public_environment_sampling_stops_before_expired_work|The public sampling loop must respect its allotted time.|Require an expired deadline to fail before shader calls.
public_environment_sampling_rejects_unqualified_orders|A caller must not silently expand the numerical workload.|Reject orders outside the declared finite quadrature sequence.
''')

CHECK_GUIDES[('tests/python/test_principled_direct_images.py::test_saved_full_head_image_rechecks_hashes_and_linear_errors', 'Saved image integrity and linear agreement')] = dict(
    purpose='Verify that the actual saved full-head pixels still agree within the declared tolerance.',
    method='Hash-check the report and binary arrays, recheck the supervised worker identity and limits, then recompute linear colour metrics against the immutable mask.',
    limits='Direct light and shared geometry only; replayed measurements, not new physics or environment/CUDA/M1 acceptance.')

_add('python/test_principled_bundle', 'Exact bundled CPU float64 profile with unchanged 32-entry query limits. Integrity and loading are not a uniform-domain proof, renderer workload, CUDA or export qualification.', '''
bundle_has_exact_profile_and_no_experimental_opt_in|Ordinary applications should use the validated table without research overrides.|Load its exact ID, check profile and domain, and differentiate a complete principled material call.
accepted_envelope_roundtrip_and_transfer_copy_storage|Saving and reloading the same validated data must retain identity without sharing mutable storage.|Round-trip through a file and envelope, copy to the same lane, compare values and verify distinct tensor storage.
evidence_label_cannot_grant_or_remove_exact_data_admission|A user's evidence label must not decide whether a table is physically accepted.|Change only the label, then change the generator, checking accepted identity versus explicit experimental loading and transfer.
each_identity_mutation_rejected_before_tensor_allocation|Changed data or provenance must not impersonate the accepted table.|Mutate each model, format, generator, domain or payload identity and intercept tensor allocation to prove early rejection.
unknown_bundle_never_reads_or_substitutes|Mistyped names and path-like IDs must fail rather than select another table.|Intercept package-data access and reject unknown, case-changed, latest and traversal-like IDs.
bundle_unsupported_lane_rejected_before_file_access|A CPU float64 result must not silently become a GPU or float32 support claim.|Request unsupported precision or device targets and verify refusal before reading data.
bundled_file_failure_is_closed_without_fallback|Missing or corrupt installed data must produce an explicit error, not a fallback.|Simulate absent, damaged and oversized files and verify no tensor allocation or substitution.
accepted_queries_need_no_further_resource_io|The lighting loop should use already loaded constants without reopening files.|Block filesystem reads after setup, evaluate both losses and check finite derivatives.
installed_bundle_bytes_match_wheel_and_have_no_dependencies|The tested table and code must be the bytes actually shipped in the wheel.|Compare every installed package member with the selected artifact, verify the fixed data hash and reject mandatory dependencies or development files.
source_archive_preserves_exact_bundle_and_catalog|Source distributions must retain the data needed for reproducible installation.|Inspect the archive without extracting it and check both data hashes, catalog inclusion and newline-preserving Git attributes.
bundled_queries_reproduce_validated_public_values|Packaging must preserve the public numerical values that were physically validated.|Verify the frozen report digest, then require bit-identical scalar replay of 80 directional and 16 average losses with the installed bundle; no new holdout claim.
''')

_add('python/test_principled_packaging', 'Development exporter safety checks, not runtime physics or automatic promotion. Only the exact frozen report chain may produce the already validated package data.', '''
packaging_refuses_changed_reports_before_parsing|Unverified measurements must not authorize bundling a table.|Change a report and require hash rejection before interpreting its contents.
packaging_never_overwrites_existing_output|An exporter must preserve existing package artifacts and stop on collisions.|Select an occupied path and verify its bytes survive unchanged before any source access.
''')


_add('python/test_public_budget_probe', 'Harness tests with synthetic controls, not measured physical table acceptance. Real calculations run separately under a fixed watchdog and memory cap.', '''
guard_cohort_counts_and_conditional_boundary|The reference screen must cover the promised number of fixed cases.|Count unique points including both sides of the generator allocation boundary at each roughness.
guard_runs_both_orders_and_preserves_every_endpoint|Every reference comparison must use both predeclared integration resolutions.|Intercept the paired evaluator and count calls, saved rows and zero-reference handling.
guard_rejects_nonphysical_reference_values|Invalid reference values must never produce a passing acceptance result.|Inject NaN, infinity, negative loss and excessive loss while preserving explicit invalid rows.
guard_cannot_pass_incomplete_evidence|Missing results must not be mistaken for successful validation.|Submit empty, short and oversized row sets and require rejection.
guard_failure_is_preserved_without_relaxing_tolerance|The declared error threshold must remain fixed when a comparison fails.|Inject a controlled discrepancy and check the exact unchanged relative-error limit.
guard_checks_deadline_before_each_reference_call|The reference worker must obey its cooperative time budget.|Expire the deadline before computation and verify that no reference call occurs.
preflight_rejects_changed_reference_before_fixture_loading|Numerical evidence must use the frozen reference implementation.|Modify a temporary reference file and require a hash mismatch before fixture construction.
physical_stage_requires_successful_supervised_predecessor|Later physical checks must not bypass a missing earlier gate.|Require the expected successful supervised predecessor before each subsequent stage.
public_adapter_never_substitutes_private_evaluation|Physical measurements must exercise public functions, not the research candidate.|Track public directional and average calls and their gradients; use the separate oracle only for numerical integration.
''')

_add('python/test_public_budget_physical', 'Finite prospective CPU float64 screen, not continuous-domain proof, CUDA, complete renderer acceptance or automatic resource admission. Only explicit supervised workers load saved physical fixtures.', '''
public_directional_values_and_gradients|Check how much light is lost at each viewing angle and whether fitting gets accurate derivatives.|Compare 80 public loss values and three first derivatives with independent 512-order integration and a coarser table.
public_mean_values_and_gradients|Check the average loss used to allocate reflected energy across directions.|Compare 16 public means and two derivatives with streaming independent integration and separate scalar-curve quadrature.
public_furnace_energy_balance|A surface in uniform white lighting must not invent energy or darken an ideal white reflector.|Measure 192 combinations using independent specular integration and the public loss-to-average ratio, retaining refinement differences.
saved_supervised_stage_evidence|Make the saved physical results inspectable without repeating expensive calculations.|Read each complete worker report, verify identity and limits, and recheck every numerical threshold before publishing its rows.
''')


_add('python/test_shading_public_lobes_lights', 'Public eager CPU lighting fixtures. No custom-light integration, shadows, material, CUDA, compile/export or complete M1 qualification is implied.', _LIGHT_LAYOUT_GUIDES)

_add('python/test_classic_material_runtime', 'Public classic and tagged-light CPU composition. Classic remains empirical; these tests do not admit a principled resource, qualify CUDA/export or prove independent rasterization or real-model M1 fitting.', '''
classic_grey_rgb_sharing_keeps_independent_scene_gradients|Grey and RGB material parameters should share exactly as documented.|Compare batched calls with independent scenes and their shared colour, highlight and exponent gradients across every allowed reflectance layout.
inactive_classic_extreme_finite_vectors_have_zero_gradients|An inactive surface must not generate invalid derivatives while normalizing tiny vectors.|Use extremely small finite opposite directions and a large exponent, requiring exactly zero values and all-input derivatives.
tagged_integration_all_input_gradcheck_and_horizon|Lighting should apply the correct measure while remaining usable in fitting.|Finite-difference every continuous input for delta and solid-angle packets, then check the declared zero horizon derivatives.
classic_demo_compares_public_images_without_regenerating_reference|The comparison page must evaluate public code against a separate reference.|Give the producer independent normal-incidence images and require eight successful measured comparisons without a persistent corpus dependency.
''')


_add('python/test_shading_public_complete', 'Mixed classic/principled public contract obligations; each test outcome is separate. Classic gradcheck does not depend on a principled resource. Synthetic resources check algebra and graph wiring, never conservation or physical admission. No GPU/export or allocator-peak acceptance.', _COMPLETE_GUIDES)

_add('python/test_principled_runtime', 'Public installed-wheel CPU float64 experimental material/resource tests. Synthetic curves verify interfaces and mathematical self-consistency, not GGX physical accuracy. No accepted resource, CUDA, export, full renderer or real-model M1 claim.', '''
budget_unsupported_lane_rejected_before_allocation|Unsupported device or precision must not silently cast or allocate an unqualified resource.|Intercept tensor construction while requesting float32, CUDA and meta targets through creation and transfer.
budget_metadata_storage_and_constructor_contract|Applications need honest resource limits and independent constant storage.|Reject direct construction and mutation; verify copied byte accounting, the experimental lane and a finite query-entry limit.
budget_parser_rejects_invalid_numbers_and_nested_data|Untrusted metadata must produce clear errors without leaking parser or conversion failures.|Reject boolean domains, gradient-bearing tensors, enormous integers, nonfinite JSON numbers and excessive nesting.
budget_static_option_types_are_explicit|A boolean must not silently become a byte count or an integer become a mode flag.|Require TypeError for wrong workspace count, backward flag and query-limit types.
principled_empty_and_tiny_inactive_keep_all_gradients|Empty or unlit surfaces must remain safe in a fitting graph.|Check zero-sized point/light axes and extremely tiny inactive vectors for exact zero values and every input gradient.
principled_preflight_counts_expanded_channels_before_allocation|A scalar query estimate must not conceal the larger complete RGB material workload.|Supply a limit that fits one query but not expanded channels, and intercept allocation before the expected rejection.
principled_reuses_view_mean_and_rebuilds_fitting_graph|Reuse immutable constants, not stale parameter-dependent losses between fitting steps.|Observe one mean query per active point across lights, then update albedo and evaluate a fresh decreasing result.
principled_and_custom_facades_match_backend|The friendly namespace should compute the same answer as explicit PyTorch calls.|Compare principled and custom Lambert facade outputs with their backend functions on identical inputs.
public_budget_mean_matches_independent_segment_quadrature|The budget mean must really integrate the curve used for directional shading.|Compare analytic moments and their derivatives with independent per-segment Gauss integration at three parameter pairs.
native_texture_roughness_fresnel_to_principled_gradcheck|Fitting should reach roughness and reflectance maps, not only a surface colour.|Finite-difference source texels and UVs through native bilinear sampling and the complete experimental material.
complete_material_demo_is_finite_and_labels_experimental_budget|A convincing picture must not disguise synthetic resource data as physical acceptance.|Check five finite sphere images, the explicit experimental label and linear built-in/custom Lambert equality.
''')

_add('python/test_shading_public_lobe_layouts', 'Public eager installed-wheel CPU fixtures, not a complete material, native C shading, device matrix or compilation/export qualification. Formula controls retain quadrature tolerances; classic highlights are not energy-conserving.', '''
public_lobe_strided_batches_match_values_and_shared_gradients|Several surfaces must not mix their values or lose shared fitting parameters.|Compare batched noncontiguous calls with separate scenes in both precisions, including derivatives, repeatability and unchanged inputs.
public_lobe_empty_calls_keep_connected_gradients|An empty selection should be a valid differentiable operation.|Check the empty output layout and connected zero gradients for every elementary function with and without batching.
public_scalar_sharing_does_not_expand_query|Shared parameters must obey the documented point and scene dimensions.|Exercise scalar, per-point and explicit batch layouts against a closed-form Blinn highlight and check derivative shapes.
public_rgb_sharing_keeps_explicit_channels|RGB reflectance sharing should never invent colour channels or extra scenes.|Compare each allowed shared RGB layout with the independent Schlick polynomial and verify parameter gradients.
public_lobe_rejections_name_the_input_category|Invalid requests should fail clearly instead of broadcasting into the wrong calculation.|Reject wrong shapes, precision, device, nonfinite values and domains, including invalid shared parameters in empty calls.
public_ggx_lost_squared_divisor_and_smith_mask_are_explicit|Tiny roughness and unlit surfaces must not quietly produce invalid arithmetic.|Reject an underflowed positive GGX divisor and demand exact finite zero Smith values and gradients on inactive boundaries.
public_schlick_endpoints_have_polynomial_derivatives|Fitting at an endpoint must follow the stated polynomial rather than a clamp.|Check analytic derivatives at zero and one cosine with RGB reflectance endpoints.
public_ggx_projected_area_is_normalized|The distribution of tiny surface facets must integrate to the expected projected area.|Numerically integrate three roughness settings against the independently known unit result at a fixed grid and tolerance.
public_microfacet_product_has_independent_roughness_one_furnace|The public factors must combine with correct units and normalization.|Integrate their product under a uniform environment and compare with the analytic value one minus log two.
public_shade_facade_is_lazy_and_matches_adapter|Plain gffx should stay lightweight while exposing the new shading functions.|Use a fresh process to check inert facade imports, then compare a facade operation with its explicit PyTorch counterpart.
public_lobe_demo_calls_installed_functions_and_labels_scope|The public demonstration should actually use the functions an application can call.|Observe calls to all five installed public functions, check its finite image and angular samples, and require a public-elementary scope label.
''')

_add('python/test_shading_image_bundle', 'Storage and corruption fixtures use artificial arrays, not rendered numerical answers. Hash integrity is not source authenticity, physical admission or public shader acceptance. Stored snapshots are data and are never executed.', '''
bundle_roundtrip_is_exact_development_and_manifest_last|Saved references must retain exact values without claiming a higher qualification.|Round-trip every typed field, preserve the parent development status, and reject identical publication without changing existing bytes.
bundle_rejects_changed_or_incomplete_evidence|Incomplete or altered evidence must never silently become a usable reference.|Damage the manifest, array payload, registered parent or completion marker and require an explicit rejection.
bundle_rejects_rehashed_malformed_metadata|A correct file hash cannot make a malformed format safe.|Rehash invalid schemas, paths, sizes, fields, shapes, dtypes, offsets and source/license claims, then test semantic validation.
bundle_rejects_rehashed_invalid_binary_values|A well-sized binary file may still contain invalid values.|Inject NaN or a non-boolean mask byte, repair its integrity hash, and require value-level rejection.
writer_validates_arrays_before_creating_directory|Bad inputs should not create a misleading partial experiment.|Submit a nonfinite array and verify that no new path is created.
binary_codec_is_little_endian_and_never_uses_pickle|Stored arrays should have portable, explicit bytes rather than executable serialization.|Compare two float64 values to an independent little-endian struct encoding and reject an invalid boolean byte.
bundle_rejects_unsafe_explicit_paths|An explicit artifact path must remain within the named corpus root.|Reject traversal, drive-qualified, backslash and reserved-device paths before reading any artifact.
''')

_add('python/test_shading_image_probe', 'Development generator and finite presentation controls. Display colour conversion does not change raw float64 evidence. Declared process limits are separate from a measured worker report; no public shader, live server or principled admission is implied.', '''
display_conversion_is_srgb_and_does_not_change_linear_evidence|The picture shown to a person must not replace the linear data judged by tests.|Check known sRGB values, explicit clipping, unchanged input arrays and rejection of nonfinite input.
reference_contact_sheet_is_finite_labelled_and_not_a_residual|A reference-only demonstration must not look like an implemented candidate comparison.|Check sixteen numbered panels, bounded PNG dimensions and explanations without fabricated candidate or residual images.
probe_refuses_a_mismatched_installed_wheel_before_generation|Generation should identify the actual installed artifact, not trust a version string.|Require an existing exact wheel before any reference generation; the supervised worker also compares installed file bytes with it.
probe_checks_wheel_bytes_not_just_the_version_string|A plausible wheel filename must not conceal altered or missing installed code.|Supply archives with changed package bytes, omitted executable files or duplicate members and require a mismatch before generation.
probe_declares_existing_bounded_supervisor_and_no_automatic_services|Image setup must have a declared CPU, memory and time boundary.|Inspect fixed limits and the supervised worker entry point without starting a server or claiming a measured peak from configuration.
comparison_refuses_underflow_instead_of_claiming_zero_error|A tiny nonzero difference must not be reported as a perfect match.|Use a representable error whose square underflows and require an explicit reporting failure rather than zero MSE.
image_demo_publication_uses_supported_provenance_and_retains_full_identity|The saved image demonstration must use the dashboard protocol without losing its detailed origin.|Validate the start event with the real provenance guard, preserve the bundle and full producer in its evidence record, and require all sixteen images and a successful finish.
''')

_add('python/test_shading_image_reference', 'Reference tooling with shared installed gffx camera/raster inputs, not an independent rasterizer or public shader acceptance. Requires a camera-capable installed wheel; the default older environment is not a source/binary fallback. No principled resource or live demonstration acceptance is claimed.', '''
image_geometry_fields_are_finite_unit_and_masked|Image references need valid geometry and an explicit distinction between head and background.|Check bounded coverage, finite float64 fields, unit normals/views and exact background sentinels.
camera_projection_and_world_roundtrip_have_independent_pinhole_answers|A reconstructed surface point must land on the pixel where it was sampled.|Compare camera output and interpolated world positions against independently written pinhole projection at pixel centres.
perspective_weights_differ_from_affine_and_reconstruct_attributes|Screen-space interpolation must not distort surface attributes under perspective.|Require normalized corrected weights, a detectable difference from affine weights and correct world-position reconstruction.
constant_vertex_and_texture_sources_equal_the_explicit_field|The same constant colour should not depend on whether it came from vertices or a texture.|Compare both native source paths to the explicit RGB field across covered pixels.
every_visible_texture_sample_matches_independent_bilinear|UV seams and image-row orientation must survive raster-to-texture sampling.|Check every visible gradient/checker sample against the separate scalar bilinear sampler and require both sides of the UV seam.
direct_images_match_independent_vector_closed_form|The scalar lighting reference needs an independently expressed image-level numerical control.|Compare all four colour sources under diffuse and classic-highlight light against a separate vector equation on identical geometry buffers.
unit_environment_lambert_images_preserve_albedo_and_background|Uniform incident radiance has an exact diffuse-only image answer.|Check albedo on positive-view covered pixels and exact black background, without claiming complete material environment integration.
image_environment_controls_are_bounded_independent_points|Check the analytic diffuse image at a small declared set of actual surface points.|Run independently converged hemisphere quadrature at eight deterministic visible pixels and enforce the total sample ceiling.
image_comparison_reports_denominators_errors_and_exact_match|A comparison report must say how many pixels were judged and how errors were measured.|Use small exact and known-offset images to verify channel counts, signed error, MAE, MSE and PSNR, including JSON-safe exact-match reporting.
wrong_image_negative_controls_cannot_pass|An apparently plausible but incorrect image must not pass acceptance.|Perturb background, channel ordering or light intensity and require the strict comparator to reject the result.
invalid_comparison_inputs_fail_before_metrics|Malformed images or an empty mask cannot yield meaningful accuracy numbers.|Reject nonfinite values, shape/dtype mismatches and invalid or empty masks before computing metrics.
repeated_buffer_build_is_identical_and_profile_limits_are_fixed|A reference must repeat on the same environment and stay within its declared workload.|Compare two complete buffer builds and refuse a resolution increase outside the fixed profile.
''')

_add('python/test_shading_public_images', 'Public classic CPU image comparison, with earlier missing-API failures retained as history. Shared geometry isolates shading rather than independently testing rasterization. No principled budget admission, Cycles equivalence or real-model M1 claim.', '''
public_classic_head_image_matches_independent_reference|Verify the public material and light APIs on controlled synthetic-head images.|Compare four albedo source passes with diffuse and classic specular settings against an independent scalar lighting reference, preserving exact background and fixed linear tolerances.
''')

_add('python/test_shading_environment_reference', 'Independent scalar reference, not a public shader, qualified image or physical-resource admission. Exact roughness-one budgets qualify special cases only. Shifted-grid/refinement agreement is empirical, not a proof against unsampled features; deadlines are cooperative for trusted callbacks.', '''
gauss_rule_has_positive_weights_and_exact_polynomial_moments|Check the numerical ruler before measuring a material with it.|Require positive interior quadrature nodes and exact known polynomial integrals at four orders.
constant_environment_preserves_units_sign_and_near_zero|Uniform light must retain the correct brightness, RGB sign and tiny values.|Compare Lambert output to albedo times radiance, including negative, zero and near-zero light, with at least three grid levels.
all_sh_bands_match_analytic_lambert_at_tilted_normals|Rotating a surface must not scramble environment coordinates or SH bands.|Compare independent hemisphere integration against the closed band-two diffuse convolution at three normals.
classic_overreflection_remains_visible|A measuring tool should expose an empirical material that reflects too much light.|Require the independently known 25/24 result, not a clamp to one or a false conservation claim.
complete_principled_exact_budget_matches_independent_moments|Check complete material integration without a circular approximate-budget oracle.|Use exact roughness-one losses under constant and sloped environments, and require agreement with analytic moments while rejecting the diffuse-only shortcut.
refinement_requires_two_transitions_and_is_repeatable|One fortunate pair of grids is not enough to declare convergence.|Inspect three consecutive grid histories, two passing transitions, work counts and exact same-machine repetition.
shifted_azimuth_breaks_false_coarse_grid_agreement|Samples can repeatedly miss the same bright and dark stripes.|Use an oscillating nonnegative test BRDF that fools early grids; require shifted-grid disagreement and an unresolved reference instead of success.
work_budget_stops_before_starting_an_unaffordable_level|Reference accuracy must not silently increase the allowed workload.|Count actual callback invocations and refuse the next complete pair of grids before overspending.
cooperative_deadline_withholds_unfinished_reference|A timed-out calculation must not become a usable last-known answer.|Advance a fake clock inside one callback and require an explicit timeout with counted partial work and no accepted result.
invalid_inputs_and_nonrepresentable_results_fail_explicitly|Bad geometry, light or arithmetic should fail clearly rather than poison a reference.|Reject malformed shapes, nonfinite fields, negative BRDFs, overflow and enlarged work controls.
reference_is_independent_and_standard_library_only|The reference must not accidentally call the same shader it is supposed to check.|Inspect imports and restrict the numerical module to math and time.
huge_integer_coefficients_raise_representability_value_error|An enormous coefficient needs the same clear representability error as other invalid numeric inputs.|Pass a ten-thousand-digit integer and require ValueError instead of a leaked conversion OverflowError.
input_coefficients_are_copied_and_callback_directions_are_unit|A caller mutation must not change the environment halfway through the integral.|Mutate the original coefficients in a callback, check unit positive-hemisphere directions and compare the unchanged light answer.
environment_demo_retains_actual_histories_and_failed_alias_case|The teaching demonstration must show measured behavior, including failure.|Run three bounded cases, inspect their actual histories and analytic answers, and require the alias case to remain labelled unresolved.
''')

_add('python/test_corpus_registry', 'Local development registry only. Integrity hashes are not physics/quality acceptance or protection against a hostile root owner replacing all provenance. Temporary test directories are not production corpus registration; simulated reparse checks do not require Windows link privileges.', '''
missing_entry_names_unregistered_id|A test must distinguish missing registration from an absent graphics operation.|Load an unknown corpus ID and require an explicit error naming the unregistered entry.
registration_roundtrip_preserves_development_and_profile_boundaries|An uncommitted test asset must not become a published or qualified reference.|Register and verify copied bytes, require development opt-in and reject an unmeasured rendering profile.
duplicate_registration_never_overwrites_files|Saving the same entry twice must not destroy existing evidence.|Snapshot every file, attempt registration again and compare all bytes after the expected error.
registry_rejects_invalid_index|The entry index must be unambiguous and contained within its root.|Reject duplicate IDs or JSON keys, unsupported schema, missing digest and a traversal path.
manifest_rejects_invalid_semantics_after_hash_verification|A correct hash cannot make contradictory metadata valid.|Rehash deliberately malformed manifests and require license, source, file, schema and admission checks to reject them.
manifest_rejects_unsafe_file_paths|An asset manifest must not reach outside its entry directory.|Reject parent traversal, absolute and drive paths, Windows streams and backslash ambiguity before opening artifacts.
tampered_bytes_are_not_accepted|Stored reference content must remain the content that was registered.|Append bytes to a manifest or scene and require hash or size mismatch errors.
oversized_registry_is_rejected_before_json_parse|Reject an oversized index before allocating decoded JSON structures.|Create one byte over the metadata cap and intercept parsing to prove the size check comes first.
interrupted_index_publication_leaves_no_registration|A failed publication must not advertise a partially written entry.|Simulate atomic-index replacement failure, retain the unregistered directory and require lookup to remain missing.
existing_registration_lock_is_never_taken_over|A concurrent or crashed writer must not be silently overridden.|Create an existing lock and verify registration refuses it without changing its contents.
loader_never_executes_copied_generator|Copied source is provenance data rather than executable loader input.|Substitute a coherently rehashed source snapshot that would raise if executed, then verify ordinary loading succeeds.
nonfinite_json_numbers_are_rejected|Overflowed JSON numbers must not enter a reproducible scene description.|Reject explicit numeric overflow as well as inconsistent tampered scene metadata.
reparse_paths_are_rejected_without_requiring_link_privileges|Windows junction or link metadata must not redirect file verification.|Simulate a reparse attribute at the scene path and require rejection before the file body is consumed.
''')

_add('python/test_shading_public_edges', 'Public entry-point fixtures; missing imports block later assertions. Synthetic curves do not prove physics. Named constructor interception is not an exhaustive allocator audit; no GPU, export or live publication claim.', '''
public_resource_knot_uses_declared_right_derivative|A fitting step at a table knot must use the promised derivative.|Use a synthetic log-linear angular curve with exact values and slopes, including the horizon-to-log junction.
public_resource_plateau_limiter_keeps_zero_slope|A flat lookup segment should not invent a fitting direction.|Check interior and endpoint plateau values with zero derivative on a turning synthetic curve.
workspace_rejection_precedes_query_tensor_allocation|Reject work before allocating its query intermediates.|Intercept named tensor constructors and stack/concatenation after resource loading, then supply one byte less than the required estimate.
normal_incidence_roundoff_keeps_value_and_tangent_gradients|Roundoff at aligned directions must not break a valid normal-incidence material.|Check the classic closed form and vanishing tangent gradients with nonunit aligned vectors in both floating precisions.
roundoff_policy_does_not_clamp_supplied_lobe_cosines|Internal normalization allowances must not hide invalid caller inputs.|Supply elementary Schlick cosines outside the unit interval and require domain errors.
custom_light_batched_strided_integration_gradcheck|Custom lights must preserve gradients and scene separation for strided batches.|Gradcheck point positions, normals and shared power for delta and solid-angle packets, then compare independent scene calls.
''')

_add('python/test_synthetic_face_fixture', 'Only a controlled first-party test asset. In-memory generation and serialization do not register a corpus, qualify rendering or replace real FLAME/consumer fitting. No persistent asset or live demonstration is created by these tests.', '''
fixture_matches_frozen_recipe_and_counts|The test head should be predictable rather than an artist-dependent input.|Check the fixed vertex/triangle counts, exact poles, analytic nose point and finite coordinates.
fixture_is_connected_closed_outward_and_nondegenerate|A bad test mesh could falsely blame the renderer or fitting engine.|Independently check face areas and orientation, opposite edge incidences, connectedness and Euler characteristic.
corner_uvs_keep_welded_geometry_and_unwrapped_seam|The texture seam must not split the geometric surface.|Check shared geometric indices with distinct zero/one UV endpoints and bounded per-triangle UV spans.
texture_patterns_and_vertex_fields_are_explicit|Known diagnostic colours make sampling mistakes visible.|Check texture dimensions, constant and gradient values, checker transitions and vertex-colour counts.
canonical_serialization_is_repeatable_independent_and_finite|Repeated generation should produce stable local evidence without shared mutable state.|Compare canonical bytes and hashes, roundtrip JSON, mutate a separate instance and reject nonfinite output.
fixture_carries_declared_camera_light_and_material_profiles|A test scene needs reproducible viewing and lighting settings, not just geometry.|Verify the fixed camera, linear output, light and two material profiles while keeping the unadmitted resource ID explicitly absent.
''')

_add('python/test_shading_independent_evaluator', 'Independent scalar test reference, not a public runtime implementation. Synthetic budgets validate formulas only. Native sampler comparison is an installed CPU lane, not new-wheel, GPU, physical-resource, Cycles or image acceptance.', '''
independent_loss_curve_linear_value_mean_and_domain|The reference lookup and its average need known correct answers.|Check a two-node linear loss curve and its analytic weighted mean, then reject invalid domains.
independent_log_curve_knots_and_integral|Validate curved interpolation and integration without trusting the candidate.|Compare log-linear values and the independent segment integral with a hand-integrated closed form.
independent_material_closed_forms_reciprocity_and_mask|The reference BRDF must be right before judging another shader.|Check oblique roughness-one principled algebra, reciprocity, inactive zero output and classic normal incidence.
independent_bilinear_sampler_preserves_orientation_and_repeat_seam|Reference sampling must agree with explicit texel and UV conventions.|Use a two-by-two colour map to check top/bottom, midpoint interpolation and periodic seam equality.
reference_imports_only_standard_library_and_limits_work|The independent reference must not call the implementation it checks.|Inspect imports for standard-library-only dependencies and reject inconsistent endpoint losses.
reference_direct_light_buffer_masks_background|Reference output must apply one cosine and keep uncovered pixels black.|Shade known normal-facing buffers, check weighted colour and reject an empty covered mask.
reference_nonconstant_roughness_matches_analytic_power_field|Constant roughness tables cannot test the roughness interpolation path.|Compare a power-law roughness and quadratic log-angle field, including an independently differenced derivative.
reference_rejects_malformed_payloads|Bad reference inputs must fail instead of producing believable numbers.|Reject booleans, nonfinite loss, duplicate or oversized axes and incorrect endpoint-pair shape.
reference_payload_copy_and_nonconvergence_are_explicit|Reference provenance and integration failure must remain trustworthy.|Mutate the input after construction and deliberately exhaust refinement to require an explicit convergence error.
native_sampler_matches_independent_reference_on_fixture_patterns|The generated diagnostic textures must use the same UV convention as native sampling.|Compare all three fixture maps at texel, interior and seam coordinates through the installed CPU adapter and independent scalar sampler.
''')


_add('python/test_shading_public_lobes_lights', 'Public elementary eager CPU API fixtures; earlier missing-import baselines are retained as history. Small analytic and finite-difference tests do not establish all-domain, GPU, compiled, real-time or full-scene acceptance. No complete material or physical-resource admission follows.', '''
each_public_lobe_matches_closed_form|Verify the reflection building blocks independently of the implementation.|Compare Lambert, Blinn, roughness-one GGX, Smith and Schlick against fixed-angle formulas in float32 and float64.
public_lobe_interior_gradients|Fitting needs correct derivatives for every lobe input.|Run float64 finite-difference gradcheck on every public lobe at smooth interior values.
blinn_horizon_corner_has_declared_value_and_zero_gradients|Avoid undefined log-zero behavior at the Blinn horizon.|Check exponents zero, one-half and two against the declared corner values and zero first derivatives.
smith_horizon_masks_before_division|A grazing view or light should take the explicit zero branch safely.|Check exact zero values and gradients with one or both cosines at zero.
ggx_rejects_domain_and_lost_positive_divisor|Do not conceal invalid roughness by adding an epsilon floor.|Reject zero, negative and above-one roughness plus a positive float64 value whose fourth power underflows.
classic_blinn_negative_control_retains_overreflection|Prevent the empirical classic model from acquiring an incorrect conservation claim.|Use a bounded 4096-point midpoint integral and the independent 25-over-24 closed form; require the overreflection to remain visible.
public_sh_constant_and_signed_environment|Keep radiance-to-irradiance units and signed SH semantics explicit.|Check positive and negative constant environments in both supported floating precisions.
public_sh_directional_projection_matches_truncated_kernel|Band-two SH has predictable truncation and ringing, not exact point-light behavior.|Build the positive-z projected coefficients analytically and check front, back and the negative ringing minimum.
public_point_inverse_square_translation_and_coincidence|Point lights must preserve physical distance scaling and safe coincidence handling.|Check fourfold distance-squared attenuation, translation invariance and zero derivatives only for the invalid coincident pair.
public_directional_has_no_hidden_distance_or_cosine_factor|Directional-light outputs must remain unshaded incident weights.|Normalize two nonunit directions, retain RGB weights exactly and reject the implication of an ordinary PDF.
public_light_interior_gradients|Light fitting must reach coefficients, locations, directions and intensities.|Gradcheck SH, point and directional public calls in float64 away from singular configurations.
''')

_add('python/test_shading_public_sources', 'Public source-to-light integration fixtures. Classic is implemented; principled/resource cases remain separate and unimplemented. Native sampler contracts stay unchanged. Synthetic principled budgets cannot prove physical admission. No independent raster, Cycles, GPU or M1 claim.', '''
constant_corner_and_texture_sources_match_complete_lit_output|Equivalent source colours should remain equivalent through material evaluation and illumination.|Compare constant corner interpolation and a constant texture under classic and synthetic principled, independent normal-incidence formulas and source gradients.
nonconstant_texture_to_lit_output_gradcheck|Fitting must reach source texels and UV coordinates through the complete chain.|Rebuild a one-level native pyramid, sample a nonconstant map and gradcheck lit output under both materials.
face_corner_uv_seams_survive_public_shading|Adjacent faces may deliberately use different texture coordinates at a seam.|Interpolate two independent corner-UV sets, require different lit colours and finite gradients to both sides of the seam.
learned_texture_rebuilds_graph_for_each_optimization_step|A fitting loop must not reuse stale learned texture data or an already-consumed graph.|Run two explicit texture-gradient steps with a fresh pyramid each forward and require the second objective to decrease.
coverage_selection_keeps_background_out_of_material_inputs|Uncovered pixels are background, not a zero-roughness material.|Select covered samples before material evaluation, then composite and independently verify foreground and background values.
''')

_add('python/test_shading_boundary_contract', 'Public boundary red baseline. Assertions after missing API imports have not executed. Two-node resources are synthetic algebra controls, not physically accepted budgets. No complete scene, GPU, compile/export, allocator-peak or deployment claim.', '''
custom_parameters_validate_before_inactive_callback|Empty or back-facing work must not conceal invalid shared inputs.|Forbid callback invocation while supplying a Python number, NaN or mismatched per-point field; require the declared error category.
two_sided_batched_callback_order_and_parameter_gradients|Keep each material field attached to the right scene and point when normals flip.|Pack four active pairs in B/P/Q order, inspect the oriented normals and parameter values, and check summed gradients back to each original field.
custom_interior_direction_and_parameter_gradcheck|Learnable custom materials need gradients through normalized directions and explicit parameters.|Apply finite-difference gradcheck to a smooth positive custom formula at an interior configuration.
empty_material_still_rejects_shared_domain_error|No pixels is not permission to accept illegal material parameters.|Use empty classic/principled calls with negative albedo or zero roughness and require domain rejection.
classic_horizon_zero_gradients_in_both_dtypes|Use the same explicit horizon convention in float32 and float64.|Check exact zero outputs and connected zero gradients for every material and direction input at the light horizon.
two_node_resource_mean_and_query_gradients_have_independent_closed_form|The average loss must integrate the same angular curve used by the material.|Use a synthetic linear two-node curve with independent directional, cosine-weighted mean, cosine and Fresnel derivatives.
resource_query_shape_precision_domain_and_work_errors|Prevent hidden broadcasting, precision changes, extrapolation and ignored work limits.|Supply wrong field shape, float32 to a float64 resource, out-of-domain roughness and a zero workspace budget.
resource_same_lane_transfer_has_independent_storage|An explicit resource transfer must not mutate an existing differentiable computation.|Require a new resource, equal values and retained experimental status; mutate inspection copies and verify the original graph still differentiates.
principled_rgb_budget_matches_independent_normal_incidence_formula|Keep RGB Fresnel channels separate while combining view, light and mean losses.|Compare r=1 normal-incidence BRDF and Fresnel gradients to an independent synthetic closed form, without claiming the synthetic resource is physical.
empty_light_integration_shapes_and_shared_gradients|Zero points or zero samples should produce usable zero results, not broken gradients.|Check output shapes and zero derivatives to normals, BRDF and shared light weights in float32 and float64.
custom_light_callback_packet_is_validated|Custom lights must provide complete finite packets and consistent invalid placeholders.|Reject missing solid angles, nonzero invalid data, NaN radiance and negative quadrature weights.
''')

_add('python/test_shading_custom_contract', 'Step-4 public red baseline, small CPU float64 fixtures. Missing APIs fail without stubs or private substitutes. Analytical Lambert checks are not a conservation claim for arbitrary callbacks, nor GPU, export, rendering-speed or live-dashboard acceptance.', '''
registries_are_local_and_return_stable_bindings|Keep applications independent and avoid silently changing an existing renderer.|Create two registries, resolve a function, replace the name explicitly and check the old bound function keeps its identity and behavior.
failed_registration_keeps_existing_state|Reject collisions, reserved names and invalid inputs without damaging existing registrations.|Try duplicate, reserved, malformed and incorrectly typed registrations in both namespaces; check the original mapping is unchanged.
registry_snapshots_missing_names_and_no_callback_execution|Looking up extensions should neither run user code nor expose a mutable registry view.|Forbid callback execution, check sorted tuple snapshots and verify missing-name lookups/removals preserve state.
direct_custom_lambert_matches_builtin_and_shared_gradients|Use a custom material without registration and retain fitting gradients.|Check normalized active-pair packing, equality to zero-specular classic and summed shared-colour gradients.
inactive_pairs_never_invoke_custom_material|Prevent custom formulas from encountering undefined inactive half-vectors.|Use empty points and back-facing lights; forbid callback invocation and require correctly shaped zero outputs and gradients.
custom_material_rejects_bad_callback_output|A material extension must return the promised BRDF tensor.|Reject incorrect shape, negative or nonfinite values, mixed dtype and a dictionary instead of a tensor.
callback_exception_propagates_without_fallback|A broken extension must not quietly render with a different material.|Raise a unique callback exception and require it to reach the caller unchanged.
light_tag_applies_one_cosine_and_one_measure_weight|Keep point-light weights distinct from environment integration weights.|Use a half-cosine direction and known incident weights, check analytical results for both tags and ensure callback-owned data is unchanged.
custom_delta_matches_builtin_point_light_and_parameter_gradients|Demonstrate custom-light compatibility with the built-in inverse-square convention.|Compare a unit custom weight with a point light two units away at four times intensity, then check gradients to shared RGB power.
environment_sample_splitting_preserves_integral_and_signed_radiance|Splitting a sky patch must not double brightness or clamp signed SH values.|Compare one sample against two half-weight samples with one negative radiance channel and a closed-form integral.
bad_light_packets_are_not_silently_reinterpreted|Reject ambiguous or unsupported light data instead of guessing its units.|Test PDFs, unknown/mixed measures, negative delta weights, wrong validity dtype, zero valid direction and missing axes.
invalid_light_is_distinct_from_visibility_and_has_zero_gradients|Keep an invalid sample separate from a valid light hidden by geometry.|Check zero invalid gradients, then apply a separate visibility factor without changing the original validity mask.
custom_light_rejects_unsupported_measure_before_callback|Unsupported light measures should not execute user code.|Forbid callback invocation and request a PDF-based measure, requiring immediate rejection.
''')

_add('python/test_material_public_contract', 'Public CPU material/interface contract. Classic is implemented; principled/resource APIs remain separate missing-feature failures, never skipped or replaced by private recipes. Synthetic resources test interfaces, not physical accuracy. No CUDA/export or accepted-resource claim.', '''
import_gffx_stays_lazy|Installing graphics tools must not make plain import load PyTorch or native code.|Use an isolated subprocess resolving the same package under test and inspect loaded modules.
public_functions_have_explicit_keyword_contracts|Make material fields and research opt-ins visible rather than hidden defaults.|Inspect both public signatures, keyword-only fields, required budget and default memory limit.
facade_dispatch_matches_backend|The convenient gffx facade must preserve the explicit backend result.|Compare the public facade and torch material on the same small classic scene.
classic_shapes_and_closed_form|Keep scene, point, light and colour axes distinct.|Exercise batched and unbatched shared/point/explicit light layouts; compare front lighting to the independent normal-incidence formula and back lighting to zero.
shared_fields_and_strided_inputs_preserve_gradients|A shared colour should receive every point's contribution during fitting.|Use sliced noncontiguous normals and compare the colour gradient to two divided by pi.
empty_calls_keep_zero_gradient_connections|Empty geometry or lights should not break a fitting loop.|Check exact empty shapes and connected zero gradients for every floating input.
invalid_material_fields_fail_explicitly|Reject bad values instead of guessing their meaning or silently changing them.|Test Python numbers, mixed precision, ambiguous scene/light axes, malformed colours, zero normals, NaNs and wrong-model fields.
view_light_and_two_sided_boundaries|Show which side of a surface can reflect light and avoid undefined half-vectors.|Check front/back/horizon and opposite directions, two-sided normal flipping, finite output and inactive zero gradients.
direction_normalization_and_interior_gradcheck|Vector direction matters, while arbitrary vector length should not change the material.|Compare scaled vectors and finite-difference all three direction inputs at an interior configuration.
malformed_envelopes_fail_before_tensor_allocation|A malformed resource should fail before allocating working tensors.|Corrupt ten schema, identity, domain, shape and endpoint categories while forbidding common tensor allocation entry points.
loader_preserves_file_and_format_error_categories|Help users distinguish a missing file from corrupt or oversized data.|Use bounded temporary files for duplicate keys, malformed JSON and the eight-MiB limit; verify the missing-file error separately.
evidence_labels_cannot_grant_admission|A file calling itself accepted is not proof that the physics was tested.|Try three evidence labels; require default rejection, experimental labeling after explicit load opt-in and rejection of a nonexistent bundled ID.
principled_requires_separate_experimental_use_opt_in|Reading research data must not silently authorize its use as an accepted material.|Load a synthetic resource explicitly, require default material rejection, then permit finite evaluation only with the second explicit opt-in.
resource_roundtrip_and_inspection_are_independent_copies|Inspection must not change constants used by a live material.|Mutate caller data and inspection copies, roundtrip a local JSON file, and compare an unchanged directional loss.
synthetic_fixture_has_no_physical_admission_claim|Keep a plumbing fixture from being mistaken for a validated light budget.|Check its independently computed payload digest, unknown generator provenance and explicit synthetic evidence label.
''')

_add('python/test_material_mean_validation', 'Validation-only CPU float64 measurements of values and first partials. No higher derivatives, public API, GPU, exact allocator bound or shader speed claim. Small reference comparisons are not the full physical acceptance screen.', '''
streamed_mean_preserves_value_and_moving_view_derivatives|Keep exactly the same mathematical light average and fitting directions while releasing each view graph early.|Compare six roughness/Fresnel/coordinate combinations to the frozen full-graph oracle at the same 16-light/16-view rule, including derivatives of moving view coordinates and integration weights.
reference_graph_pressure_is_bounded_per_view_not_all_views|Explain why the old oracle can exhaust memory even when the material resource is small.|Count live saved tensor references around forward/backward; compare the old 16-view graph with streamed 16/32 views. These logical byte counts include repeated storage references and are not process RSS.
streaming_preserves_sample_order_count_and_finishes_each_backward|Ensure the memory fix does not skip samples or detach necessary derivatives.|Use a synthetic differentiable directional term that refuses a second live branch; require exact original node order/count, unchanged quadrature options, completed backward for every sample and correct value/partials.
streamed_oracle_rejects_invalid_work_before_any_directional_call|Keep the safety limits while changing only validation scheduling.|Reject oversized light/view counts and unknown coordinate methods before any directional evaluation; retain the 2097152-sample cap.
''')


_add('python/test_material_loss_tensor', 'Internal CPU float64 candidate only. Synthetic algebra and loader checks do not replace physical mean/furnace validation, which remains incomplete after a memory stop. No public API, GPU, export or exact allocation-bound claim.', '''
tensor_batch_values_and_all_gradients_match_frozen_scalar|Keep the same light response and fitting directions while evaluating a whole batch.|Compare1,8,32 values and all view, roughness and Fresnel derivatives to the frozen scalar recipe under predeclared tolerances.
shared_fields_strided_shapes_and_gradcheck|Support shared material parameters without losing summed gradients.|Use a noncontiguous2x2 field, scalar roughness/Fresnel, finite-difference gradcheck and separate directional/mean methods.
exact_mean_and_tiny_loss_stay_connected|Retain tiny remaining light even when reflected energy rounds to one.|Scale a synthetic resource by1e-25; compare means and derivatives to independent segment integration and check loss remains positive.
knot_and_horizon_side_conventions_match_scalar|Avoid changing derivative conventions at curve joins and the horizon.|Compare five endpoint/knot values and three partial derivatives with the frozen scalar evaluator.
two_node_resource_and_empty_queries|Handle minimal resources and empty fields without hiding bad parameters.|Compare a linear two-node curve and zero-size gradients; reject invalid shared roughness even with no points.
query_errors_before_curve_allocation|Reject unsupported inputs before expensive curve construction.|Inject wrong dtype, Python number, shape, NaN, domain, oversized batch and meta device while forbidding curve allocation.
workspace_limit_and_storage_inspection_cannot_mutate_resource|Make resource inspection safe and query work explicitly bounded.|Mutate copied inspection tensors, check unchanged output, and reject too-small workspace and oversized batches.
loading_roundtrip_and_caller_mutation_isolation|Keep loaded resources independent of caller-owned dictionaries.|Roundtrip a local envelope, mutate caller payload/metadata copies and require stable resource values and identity.
bad_envelopes_rejected_before_tensor_allocation|Validate artifact identity and structure before creating tensors.|Corrupt each required metadata/payload category and forbid tensor construction during rejection.
loader_byte_limit_duplicate_keys_and_missing_file|Fail closed on oversized or ambiguous files and retain ordinary I/O errors.|Test8 MiB byte ceiling, duplicate JSON keys and missing-file behavior.
queries_do_not_construct_constants_read_files_or_integrate|Keep offline resource costs out of repeated queries.|Forbid tensor construction, file opening and numerical quadrature during repeated evaluation and backward.
tensor_mean_matches_independent_segment_quadrature|Check that analytic means describe the curve being evaluated.|Compare a synthetic curve and both parameter derivatives to independent Gauss segment integration at1e-11.
probe_uses_real_batch_and_accounts_saved_graph_storage|Measure actual batch backward storage rather than disconnected scalar calls.|Run a synthetic two-point batch under saved-tensor hooks and compare distinct nonconstant storage with its planning estimate.
supervisor_can_launch_the_bounded_tensor_probe|Reuse the verified actual-worker watchdog for the new task.|Check real-interpreter command routing and unchanged explicit worker arguments without launching a numerical workload.
''')

_add('python/test_material_loss_tensor_physical', 'Existing48/60/60 cohorts are regression, not new holdouts. The historical mean memory stop is preserved; streamed validation now passes mean/furnace checks under unchanged limits. This guide is not live test evidence or public admission.', '''
tensor_directional_physical|Check directional loss and fitting derivatives against independent physics after batching.|Reuse the original512-node alternate-coordinate reference for168 triples, maintaining relative loss/refinement0.001, energy0.0005 and derivative error ratio1 limits.
tensor_mean_physical|Check the average light budget and its parameter derivatives against an independent integral.|Compare original128-light/64-view reference using per-view gradient measurements and independent segment quadrature; all36 comparisons retain their original limits and saved result rows.
tensor_furnace_physical|Ensure a white environment does not gain or lose unexplained light.|Use the original independent specular and integrated matte comparisons for396 cases under unchanged1.001 ceiling and0.001 error/refinement limits; saved runs own the result, not this explanation.
''')

_add('python/test_material_loss_cost', 'Bounded scalar CPU float64 measurement tooling, not physical accuracy, vectorized support, GPU speed or public admission. Saved fixture benchmarks are separate from these synthetic checks.', '''
fixture_hash_checked_before_parsing|Refuse changed saved grids before treating them as evidence.|Supply malformed JSON with a mismatching digest and require a hash error before parsing.
loader_rejects_unbounded_or_malformed_inputs|Prevent unbounded or malformed data from reaching tensor allocation.|Exercise byte and node limits, wrong shapes, nonfinite values and unknown fields.
storage_counts_constants_without_calling_them_resident_tensors|Distinguish packed numbers, Python overhead and actual tensor storage.|Count an exact17-number synthetic grid and separate all derived constants from object accounting; tuples contain no persistent tensors.
loop_batch_retains_all_three_parameter_gradients|Measure real batch backward work instead of disconnected placeholders.|Compare a synthetic analytic field and all view, roughness and Fresnel gradient sums; retain all point graphs until backward.
no_grad_forward_matches_backward_forward_value|Keep forward and backward timing workloads comparable.|Require both modes to evaluate the same scalar objective while no-grad returns no gradient summary.
batch_limit_rejected_before_input_allocation|Stop accidental scaling beyond the predeclared laptop workload.|Reject zero, bool and unapproved large batch sizes before allocation.
nonfinite_gradient_is_a_failed_measurement|Do not accept finite values with broken fitting directions.|Use a finite square-root boundary with nonfinite derivative and require a measurement failure.
deadline_stops_before_next_case|Enforce the declared work deadline.|Supply an expired deadline and require an explicit stop before further work.
profile_is_fixed_and_does_not_claim_vectorized_support|Keep the saved benchmark description honest.|Check sizes1,8,32, one warm-up, three samples, watchdog and explicit scalar-loop and no-generation qualifications.
timing_excludes_warmup_and_records_three_raw_samples|Prevent cherry-picked fastest-sample reporting.|Use a deterministic fake clock to check four calls, three saved durations and their median.
constructor_rejects_invalid_physical_knots|Keep physical validation after successful JSON parsing.|Supply an out-of-range loss with a valid file hash and require the frozen constructor to reject it.
working_set_monitor_reads_this_process|Verify the operating-system memory query actually works.|Read the test process working set and release its handle without stress allocation; platform-specific monitoring evidence only.
supervisor_stops_over_memory_budget_and_preserves_partial_report|Stop only the owned worker when the memory ceiling is exceeded.|Simulate an over-limit worker and verify termination, handle cleanup and a retained stopped report.
supervisor_refuses_to_overwrite_prior_evidence|Keep failed experiment history intact.|Attempt a duplicate output path and verify no worker starts and saved content is unchanged.
worker_launch_bypasses_venv_redirector_but_keeps_import_paths|Prevent monitoring the small Windows launcher instead of PyTorch.|Check use of the real interpreter with the same environment module paths; the actual experiment also compares worker and monitor process IDs.
''')

_add('python/test_testing_retention','Temporary local stores only. These checks do not measure sustained disk/RSS budgets or change live history settings.', '''
more_than_one_hundred_executions_do_not_consume_page_capacity|Ensure repeated testing never exhausts an arbitrary lifetime session count.|Save 105 completed executions of one test, paginate the history and verify there is still only one stable test page.
retention_protects_latest_failures_checks_partial_subsets_and_pins|Keep important evidence even when later suites run only some tests.|Mix latest failures, a removed named check, an active test and a persistent pin; inspect which old executions are eligible.
retention_keeps_newest_unprotected_history_not_a_total_run_cap|Check the history setting means extra older executions, not total page capacity.|Keep two superseded executions plus protected current evidence and identify the exact older candidates.
retire_is_transactional_and_preserves_page_identity_and_retired_ids|Prevent partial metadata cleanup and accidental reuse of old evidence URLs.|Reject a mixed protected/unprotected selection atomically, retire an eligible execution, then verify stable creation dates and a tombstone.
active_execution_capacity_and_metadata_byte_budget_are_separate|Bound unfinished sessions without limiting how many completed sessions can ever exist.|Fill one active slot, reject another, explicitly finish it and admit a replacement under a separate metadata budget.
invalid_retention_policies_fail_before_database_creation|Reject unusable configuration before writing a store.|Exercise negative counts, booleans and undersized storage budgets.
latest_coverage_and_demo_evidence_survive_retention|Keep historical measurements without presenting them as new results.|Protect the latest measured coverage and demo, while the newer unmeasured coverage page stays unmeasured.
history_preview_token_rejects_changed_protection_and_prunes_exact_artifacts|Require cleanup to match the preview and prevent resurrection.|Pin evidence after preview to invalidate it, then preview again and remove only eligible artifacts and metadata.
retention_never_deletes_through_symlink_or_unknown_files|Refuse to erase files that the dashboard does not own.|Place unexpected content inside an eligible execution folder and verify cleanup blocks that folder and preserves the file.
metadata_exhaustion_is_explicit_and_rolls_back_evidence|Never report success when the metadata storage budget is exhausted.|Overflow a small SQLite budget with bounded test metadata; require an explicit storage error and no partially collected tests.
live_viewer_leases_block_cleanup|Do not remove artifacts while a viewer is using them.|Hold a page lease on old evidence and verify it is blocked until the lease is released.
linked_artifact_directory_is_never_followed|Keep cleanup inside the selected dashboard root.|Point an execution folder outside the root and verify preview refuses it without touching the external file; skip only if link creation is unavailable.
history_http_and_pin_controls_are_explicit|Make safe retention controls available without browser command execution.|Read history over HTTP, pin/unpin saved evidence and verify malformed writes are rejected.
''')


_add('python/test_material_loss_reference',
     'Scalar CPU float64 reference only. Finite sampled admission, not full-domain proof, a fast table or a public shader. The two loss coordinates share a mathematical identity; old oracle failures remain explicit controls.', '''
rejected_probability_has_exact_normal_view|Count the rare reflections that go below the surface even when sampling misses them.|Compare four roughnesses with the independent normal-view probability alpha squared divided by one plus alpha squared, including a one-in-a-trillion tail.
loss_has_exact_rough_limit_and_derivatives|Check direct lost-light calculations against known mathematics.|At roughness and Fresnel one, compare loss and view derivative with the exact logarithmic formula at three views.
loss_mean_resolves_both_coordinates|Ensure tiny remaining-light budgets converge without subtracting nearly equal totals.|At six near-perfect-reflection pairs compare 64/32 and 128/64 quadrature using sinh and log slope coordinates; require the unchanged 0.1 percent relative residual limit and parameter-gradient refinement.
loss_mean_exact_control_and_explicit_limits|Keep mean loss correct on a known case and computation explicitly bounded.|Compare the exact roughness-one mean, reject unknown coordinates, and reject requests beyond 2097152 light samples.
loss_keeps_parameter_gradients|Keep the new integration coordinates inside autodiff.|Compare automatic roughness and Fresnel derivatives against finite differences through moving slope and view bounds.
rejected_tail_matches_direct_density_integral|Independently check the analytic rejected-light integral.|Integrate the visible-normal density numerically over a separately constructed inverse-slope interval at twelve points, then compare with the cancellation-resistant antiderivative. The physics is shared, but the integration is independently evaluated.
loss_complements_existing_energy_and_partial_gradients|Compare old reflected-light and new lost-light calculations where the old comparator is resolved.|Require their sum to be one and their partial derivatives to cancel at fourteen overlap configurations and three Fresnel values.
extreme_grazing_loss_refines_without_using_unresolved_old_gradient|Check the new reference at the two cases where the old gradient is unreliable.|Compare 256/512-node loss and partial derivatives across two coordinate schemes at cosine one millionth. This is cross-coordinate consistency, not independent physical-model agreement.
old_extreme_grazing_gradient_fails_exact_rough_control|Preserve evidence that a visually plausible old energy can have a wrong gradient.|Use the exact roughness-one derivative to require the old 512-node gradient to fail and the new loss gradient to pass the original tolerance.
old_visible_normal_sampler_misses_known_small_tail|Explain why the previous independent answer was not truth for a tiny remainder.|At normal view and roughness 0.001, show that the old finite grid counts no rejected rays although the exact probability is positive; verify the analytic tail counts them.
loss_preserves_legacy_mean_failure_as_negative_control|Keep the original failed mean calculation reproducible.|Require the old subtraction-based mean at roughness 0.003 and Fresnel one to retain its more-than-0.1-percent refinement discrepancy.
tiny_loss_gradient_matches_relative_finite_difference|Check tiny roughness derivatives without hiding them behind a large absolute tolerance.|Compare two relative finite-difference steps at three sharp roughnesses with 0.1 percent relative gradient tolerance; verify the Fresnel derivative against its exact affine slope for the quadrature.
loss_rejects_invalid_inputs_without_changing_domain|Reject invalid or unrepresentable requests without changing the material.|Check excluded view endpoints, zero or underflowing roughness, invalid Fresnel, wrong dtype and boolean resolution; no clamps or mirror substitutions.
residual_study_retains_legacy_and_adds_direct_loss|Keep the demonstration honest when showing the repair.|Verify all six pairs retain four old methods and add both direct-loss coordinates, with a separately labelled coordinate difference and a warning that the physics is shared.
''')

_add('python/test_material_budget_boundary',
     'Predeclared CPU float64 boundary screening, not global accuracy or a public fast material. Failed reference checks block admission; no energy clamp or larger work cap.', '''
coupled_grazing_reference_converges|Check very shiny surfaces seen almost edge-on, where two difficult limits meet.|At 48 triples with cosine proportional to roughness squared, compare 256/512-node energies and three partial derivatives, then require independently refined VNDF agreement and physical energy range. Records are saved before assertions.
residual_probe_has_explicit_methods_and_work_limits|Verify diagnostic plumbing without mistaking it for a repaired numerical reference.|Check four explicit methods on the exact roughness-one mean, reject unknown methods and excess work, and refuse gradients through the detached independent oracle.
residual_demo_publishes_unresolved_evidence|Ensure the diagnostic page can actually display disagreement without claiming a fix.|Publish a small fixture through a real temporary testing server and verify five artifacts and successful publication. This accepts the reporting protocol, not the numerical residual estimates.
tiny_mean_residual_is_resolved|Check whether the tiny amount of light left for matte reflection can safely be used as a denominator.|At the same six pairs and unchanged tolerances, use the directly integrated loss rather than subtracting a rounded near-one mean; inspect derivatives and swap furnace normalizations. Original five failures remain in saved executions and a separate negative control.
tiny_synthetic_budget_retains_signal_and_gradients|Separate floating-point arithmetic from unresolved reflection integrals.|Use binary-exact tiny budgets with an analytic value and derivative, and reject a zero mean residual inconsistent with nonzero directional residuals. This does not prove real quadrature has resolved the same tiny signal.
''')

for _test, _checks in (
    ('coupled_grazing_reference_converges', (
        ('Reference value refinement', 'Does the reflected-light estimate settle as sampling is refined?', 'Compare 256 and 512 nodes with the unchanged absolute 0.00005 limit at three Fresnel values.'),
        ('Reference partial derivative refinement', 'Do the fitting directions settle too?', 'Compare view, roughness and Fresnel partial derivatives; each other input is held fixed. Use 0.001 plus 0.002 times the finer derivative magnitude.'),
        ('Independent oracle refinement and agreement', 'Does a different integration method agree?', 'Require both independent VNDF refinement and disagreement with the sharp reference to stay within 0.00005.'),
        ('Reflected energy range', 'Does the sampled surface avoid creating light?', 'Require every recorded directional energy to lie between zero and 1.001; a finite screen is not a global proof.'))),
    ('tiny_mean_residual_is_resolved', (
        ('Positive and relatively resolved remaining budget', 'Can we trust the tiny amount of light left over?', 'Require both residual estimates to be positive and their relative difference no greater than 0.001, or 0.1 percent.'),
        ('Residual derivative refinement', 'Is the remaining budget sensitive to parameters in a reproducible way?', 'Compare mean-residual roughness and Fresnel derivatives under the existing gradient tolerance.'),
        ('White furnace under both normalizations', 'Does using either remaining-budget normalization preserve white-light energy?', 'Swap coarse and fine residual means in the coupled energy calculation; require the original 1.001 ceiling and error below 0.001.'))),
):
    for _name, _purpose, _method in _checks:
        CHECK_GUIDES[('tests/python/test_material_budget_boundary.py::test_'+_test, _name)] = dict(
            purpose=_purpose, method=_method,
            limits='Finite reference screening only. A failed earlier check can prevent later checks from executing; absent evidence is not a pass.')

_add('python/test_material_loss_cell',
     'Six local scalar CPU float64 cells only. No complete angular resource, integrated mean, furnace, public material or performance acceptance.', '''
log_loss_cell_exact_field_gradients_and_independent_storage|Keep a tiny positive light budget and its derivatives when reflected energy rounds to one.|Interpolate an exact log-bilinear synthetic field, run finite-difference gradcheck, mutate caller-owned knot tensors and check the cell remains unchanged.
loss_cell_rejects_invalid_data_and_outside_queries|Reject unsupported cells rather than hiding extrapolation or clamping roughness.|Check axis order, shapes, float64 storage, positivity, Fresnel endpoint ordering, explicit cell boundaries and generation resolution caps.
loss_cell_demo_publishes_finite_comparisons_without_acceptance_claim|Make the local interpolation experiment visible without claiming a new renderer.|Publish a small fixture through a real temporary server and verify five artifacts, a completed demo and the explicit admission disclaimer.
local_log_loss_cell_reference_value_gradient_and_refinement|Test whether log-space interpolation can approximate sharp-surface light loss locally.|At six predeclared centers and three Fresnel values compare two cell widths with the repaired log-coordinate reference, retaining relative-loss, energy and three-partial-derivative limits. Save measurements before assertions.
''')

_add('python/test_material_loss_resource',
     'Experimental CPU float64 resource on declared roughness [.001,1]. Exact integration of an approximation is not proof that the approximation matches physics. No public shader, speed or full-domain admission.', '''
endpoint_loss_limits_refinement_and_independent_values|Include exactly straight-on and edge-on views instead of extrapolating through a logarithmic singularity.|At four roughnesses compare derived endpoint formulas at 128 and 256 nodes, approach them with the 512-node interior reference and compare independently integrated reflected light at three Fresnel values; absolute limit 0.00005.
endpoint_exact_controls_and_roughness_gradients|Check the endpoint formulas against a known exact answer and finite differences.|Require zero horizon loss for unit Fresnel, normal-view roughness-one loss log(2), and roughness gradcheck at both endpoints.
paired_loss_geometry_agrees_with_scalar_reference|Share expensive geometry between Fresnel endpoints without changing the physical model.|Compare the paired computation with the existing scalar direct-loss reference at four view/roughness pairs and three Fresnel values; use the smaller loss/energy complement without PDF normalization.
full_angle_resource_exact_mean_and_gradients|Ensure the total remaining-light budget comes from the very same curve as directional queries.|Integrate a synthetic resource independently with per-segment Gauss quadrature; compare values and derivatives to the analytic mean within 1e-11, run gradcheck and mutate caller inputs to verify storage isolation.
resource_removable_singularity_tiny_loss_and_knot_sides|Avoid NaNs when a segment has an exactly logarithmic integral and keep a declared slope at knots.|Use a cosine-to-the-minus-two synthetic curve, finite-difference endpoint and knot slopes, and loss scaled by 1e-25; require positive loss even if reflected energy rounds to one.
resource_domain_and_generation_work_guards|Reject malformed resources and prohibit excessive generation work before allocation.|Exercise node and 1.2-billion shared-sample caps, wrong tensor types, NaN, shape and physical-range violations, plus queries outside the stored roughness interval.
full_resource_held_out_directional_accuracy_and_gradients|Determine whether a practical full-range grid is accurate, not merely internally consistent.|Compare bounded-cubic 97-by-385/129-by-513 grids against alternate-coordinate 512-node loss and all three partial derivatives. Parameter labels select development (48 triples), first fresh (60, now developmental after a generation failure), or holdout2 (60, declared before the conditional allocation screen). Preserve full records and unchanged limits; later reruns are regressions, not new unseen evidence.
full_resource_mean_matches_reference_and_its_gradients|Check whether the curve average matches independently integrated physical loss.|At twelve roughness/Fresnel pairs compare the exact stored-curve mean with 128-light/64-view direct-loss quadrature and its derivatives, and separately check the curve's numerical integral to 1e-11.
full_resource_independently_integrated_furnace|Check whether the coupled surface creates light under uniform illumination.|Sweep 108 development configurations or 144 in each subsequent cohort using independently integrated specular reflection and per-segment numerical integration of candidate matte reflection. Require energy at most 1.001, white error and coarse/fine difference below 0.001. The screen label identifies the cohort; repeated execution is regression, not new holdout validation.
dense_generation_quadrature_is_resolved|Check generation integration before trusting more densely spaced interpolation knots.|At 48 or 56 endpoint values compare base280 radial/32 azimuth against512/64. Only below cosine=roughness squared/16, halve radial and double azimuth at the same work. Require 0.01 percent relative loss agreement or 1e-12 absolute for exact zero. Earlier fixed allocations failed in different regions; neither failure is erased by this repair.
resource_demo_publishes_saved_failures_without_promoting_them|Make failed numerical trials visible without turning a demo into a passing test.|Publish a synthetic failed screen to a real temporary local server; verify finite completion, artifacts and an explicit not-acceptance explanation.
original_power_resource_retains_directional_failure|Preserve the old interpolation failures as reproducible controls.|Reconstruct the original power-segment interpolation from the current experiment's knots and require its relative loss error to remain above 0.1 percent at the two known failed triples. The relevant non-grazing knots use the original allocation. Passing accepts the negative control, not the old resource; original raw tables and executions remain separate historical evidence.
''')

_add('python/test_material_loss_generation',
     'Experimental scalar CPU quadrature allocation. The threshold is a numerical work choice, not a physical domain restriction. Finite checks are not a universal accuracy proof.', '''
generation_allocation_preserves_work_and_caps|Move samples to the angular direction at nearly edge-on views without increasing the total budget.|Check ordinary and grazing allocations, equal sample products, exact threshold behavior, and rejection of orders that would violate per-axis limits.
conditional_generator_dispatches_actual_knots|Prove the generator uses the declared allocation at each actual stored knot.|Replace the expensive integral with a recording fixture on a small grid, then compare every dispatched radial and azimuth count with the rule; the fake data is not physical evidence.
allocation_boundary_and_grazing_regressions|Avoid fixing one view while introducing a numerical seam or reviving normal-view failures.|Compare both Fresnel endpoint integrals with doubled orders at five roughnesses, the previously failing grazing views, normal views and just below/on/above the allocation boundary. Every case retains the 0.01 percent relative guard.
''')

_add('python/test_material_loss_smooth',
     'Experimental bounded-cubic resource only. These algebra checks do not establish physical accuracy, runtime speed, compiled/GPU support or public material acceptance.', '''
cubic_slopes_reproduce_quadratics_without_overshoot|Improve interpolation without creating values beyond the neighboring data.|Compare nodal slopes with an exact quadratic derivative and verify the ordered Bezier control polygon stays inside each interval.
ordered_cubic_controls_handle_plateaus_turns_and_touching_fields|Keep both Fresnel endpoint curves ordered even where their values touch or reverse direction.|Use uneven knots, a flat region, turns and touching endpoint fields; verify ordered bounded controls and finite gradients through the limiter.
bernstein_integrals_match_independent_quadrature|Check the analytic curve-average weights from tiny to very wide logarithmic intervals.|Compare six widths against independent 256-node Gauss integration, positivity and the exact integral of the four basis functions summed together.
smooth_resource_exact_field_and_query_gradients_without_quadrature|Keep the fitted curve and its average mathematically consistent and differentiable.|Use an exact synthetic field, finite-difference gradcheck and independent per-segment integration; replace the quadrature function with an error and require ordinary queries still work.
smooth_resource_positive_knot_slopes_are_continuous_and_tiny_loss_survives|Remove spurious angular slope jumps while preserving tiny leftover-light budgets.|Compare derivatives from both sides of two positive knots, scale losses by 1e-25 and require positive loss/mean even when energy rounds to one. Queries outside the resource still fail.
smooth_resource_minimum_shape_endpoints_and_storage_are_valid|Keep the smallest valid resource usable at both view endpoints.|Use two roughness nodes and one linear angular segment; compare directional values and mean with exact formulas, mutate caller arrays and reject changes to frozen derived data. This reproduces the initial empty-control-array bug.
smooth_resource_constant_unit_budget_has_exact_normalization|Check that a constant budget integrates to one without a normalization patch.|Use uneven angular intervals and an exactly constant zero-Fresnel field; require unit directional values and the analytic unit mean at horizon, interior and normal views.
''')

for _test, _names in (
    ('full_resource_held_out_directional_accuracy_and_gradients', ('Directional physical accuracy','Directional gradient accuracy','Directional grid refinement')),
    ('full_resource_mean_matches_reference_and_its_gradients', ('Mean physical accuracy','Mean gradient accuracy','Exact curve mean consistency')),
    ('full_resource_independently_integrated_furnace', ('Furnace energy and white accuracy','Furnace grid refinement')),
):
    for _name in _names:
        CHECK_GUIDES[('tests/python/test_material_loss_resource.py::test_'+_test,_name)]=dict(
            purpose=_name+' for the experimental full-angle remaining-light curve.',
            method='Inspect the parent test measurement rows. Each named check enforces its predeclared value, derivative, integration or refinement limit; all sibling checks execute even if another fails.',
            limits='The parent parameter label identifies development, first-fresh-now-development, or predeclared holdout2. A repeated run is regression, never a new holdout. No full-domain proof, runtime or public material admission.')
for _name,_purpose in (
    ('Local value and residual accuracy','Keep both reflected brightness and the tiny leftover budget accurate.'),
    ('Local partial derivative accuracy','Check that the approximation supplies useful changes for fitting.'),
    ('Local cell refinement','Check that narrowing the interpolation cell stabilizes the result.')):
    CHECK_GUIDES[('tests/python/test_material_loss_cell.py::test_local_log_loss_cell_reference_value_gradient_and_refinement',_name)]=dict(
        purpose=_purpose,method='Compare all three Fresnel endpoints/interior values against the fixed limits declared before execution; retain the per-case numeric record.',
        limits='Finite local screening only. No complete table, integrated mean or material energy admission.')

_add('python/test_material_budget_sharp',
     'Scalar CPU float64 development reference, finite sampled sharp cases only. No public material, fast-table extension, whole-domain proof, CUDA/export or image-convergence acceptance.', '''
sharp_reference_has_exact_rough_limit|Check the new coordinates against a case with an exact answer.|At roughness and Fresnel reflectance one, compare three directional energies and view derivatives with the independent logarithmic formula.
sharp_reference_resolves_values_and_gradients|Resolve the narrow reflections of polished surfaces without increasing their roughness.|Compare two transformed-reference resolutions at sixty triples, retaining strict value and three-gradient refinement limits. Independently compare tail-dense VNDF values and require that comparison to converge too.
sharp_reference_agrees_with_direct_derivatives|Cross-check the new integration coordinates where the older method is reliable.|Compare values and all three derivatives against direct hemisphere quadrature at two overlapping-domain cases, using unchanged reference limits.
sharp_reference_keeps_moving_boundary_gradients|Ensure optimization can see changes to the reflection boundary.|Differentiate through mapped quadrature nodes and bounds; compare with finite differences, repeat calls exactly and verify affine Fresnel dependence. Internal gradcheck alone is not physical accuracy.
sharp_reference_approaches_mirror_without_roughness_clamp|Check that very smooth surfaces approach the expected mirror reflection.|At roughness 0.001, compare energy to Schlick reflectance at the view angle and require a finite roughness derivative. This limit example does not establish every tiny-roughness derivative.
sharp_reference_rejects_invalid_and_unrepresentable_inputs|Reject unsupported inputs and arithmetic overflow instead of silently changing the material.|Check scalar CPU float64, finite physical domains, resolution caps, roughness underflow and an unrepresentable near-horizon integration boundary.
sharp_independent_rule_retains_legacy_nonconvergence|Keep the failed comparison method reproducible while testing its repair.|Require the three measured uniform-radial convergence failures, then verify tail-dense sampling with the same node count and limits. Check normalization and invalid rule names; uniform remains the legacy default.
sharp_average_has_exact_control_and_work_cap|Verify the total directional budget and keep its integration work bounded.|Compare the roughness-one mean to an exact integral, then reject unsupported view counts and more than 2097152 light samples before constructing a graph.
sharp_average_refines_values_and_gradients|Check average energy and its fitting derivatives on polished materials.|Refine light and view quadrature at four roughness values, recording the mean and roughness/Fresnel derivatives. Retain the original reference error limits.
sharp_coupling_keeps_independent_furnace_limits|Ensure adding matte reflection does not create energy on shiny materials.|Across 108 sharp configurations combine independent VNDF specular with separately integrated coupled matte. The denominator uses 32 view nodes and the residual integral 64; retain the 1.001 ceiling and 0.001 white-error limit.
sharp_demo_exposes_surfaces_and_reference_limits|Show polished surfaces without treating their appearance as scientific acceptance.|Check the four-sphere RGB plate, sixteen energy/gradient measurements, per-tile visible variation and the explicit 512-sample preview disclaimer. High-precision scalar tests provide separate numerical evidence.
sharp_publisher_obeys_real_dashboard_protocol|Catch integration errors that numerical rendering tests cannot find.|Publish a tiny synthetic fixture through a real temporary testing server and installed dashboard client; require valid paths, six registered artifacts and successful completion. This reproduces the original rejected equals sign without repeating costly rendering.
''')

_add('python/test_material_budget',
     'Finite scalar CPU reference checks. The legacy grid remains reproducible; this does not establish uniform accuracy for every positive roughness or a public shading runtime.', '''
budget_grazing_reference_is_resolved|Check whether the measuring stick now resolves a glossy reflection seen at a shallow angle.|Compare the retained uniform angular rule with a fixed Gauss rule at two resolutions over twelve triples. Require unchanged value/gradient convergence and independent VNDF value agreement before judging any table.
budget_azimuth_choice_is_explicit_and_bounded|Keep reference refinement explicit and prevent hidden work growth or lost gradients.|Reject unknown rules, preserve the average-work cap for both rules and check the new rule against finite differences. Fixed sample coordinates are independent of material parameters.
budget_angular_demo_explains_reference_convergence|Show why changing where we sample can matter more than simply adding samples.|Compare old and new angular rules at five resolutions in one glossy/grazing configuration, with energy and two gradient curves. Check the independent energy comparison and retain the legacy nonconvergence control; not table acceptance.
''')

_add('python/test_material_budget_table',
     'Experimental CPU table on explicit roughness 0.2..1, finite acceptance fixtures only; not complete material, all-domain derivatives, public support-floor change or deployment/performance acceptance.', '''
table_held_out_values_and_derivatives|Make the approximation earn both brightness and gradient accuracy.|Compare three held-out table queries to independent differentiable quadrature, preserving fixed value, derivative and coarse/fine limits.
table_average_is_its_exact_cosine_weighted_mean|Keep directional and average energy accounting consistent.|Integrate each quadratic-cosine segment times the cosine weight with exact two-node quadrature and compare both mean values and parameter gradients.
table_keeps_the_existing_independent_furnace_gate|Prevent a fast approximation from weakening the energy requirement.|Integrate the original specular BRDF independently and the table-coupled matte term across 108 existing parameter combinations; retain the 1.001 ceiling and 0.001 white/refinement limits.
table_preserves_tensor_shapes_fresnel_linearity_and_gradcheck|Keep query coordinates connected to optimization and Fresnel interpolation exact.|Test shared-input gradcheck away from knots, affine F0 dependence, float32 output and empty layouts.
table_range_errors_never_silently_clamp|Avoid changing unsupported materials to make the table appear to work.|Reject roughness outside the candidate range, invalid cosine/F0, ambiguous shapes and dtype/device mismatches; check both cosine endpoints explicitly.
table_generation_is_bounded_and_reproducible|Keep offline generation explicit, bounded and reproducible.|Reject invalid generation sizes and injected out-of-range energy, then repeat a small higher-precision generation exactly. The original low-precision roundoff rejection remains historical evidence.
table_average_agrees_with_independent_differentiable_reference|Check that a self-consistent table mean also approximates the original integral.|Compare average values and roughness/F0 derivatives at a held-out point against nested fixed-hemisphere quadrature.
table_demo_exposes_value_and_derivative_approximation|Make approximation limits inspectable instead of showing only a smooth-looking value curve.|Check 41 roughness reference/table pairs and six grazing-angle value/gradient pairs, finite slopes, exact tensor payload size and the textbook's explicit roughness boundary explanation.
table_expanded_directional_accuracy|Find where the table gives inaccurate brightness or fitting directions, including glossy and grazing cases.|Sweep ten view cosines, four roughnesses and three Fresnel values: 120 triples, including extra grazing points. First compare two reference resolutions; only then judge the table against unchanged screening limits. Records distinguish an unresolved reference from a table failure.
table_knot_one_sided_derivatives|Check what happens when a query reaches a table boundary.|At both endpoints and two internal knots on each axis, compare autodiff with a finite difference into the declared right or final cell; also check continuity. This checks the interpolant, not physical derivative accuracy.
table_float32_values_and_gradients|Check that reduced precision preserves the same approximation and its gradients.|Compare float32 with float64 at three points away from knots, retaining values and derivatives for both directional and average budgets. Agreement between dtypes is not agreement with physics.
table_sharper_materials_are_explicitly_unsupported|Make the glossy-material coverage gap visible rather than silently rounding materials up.|Query four positive roughnesses below 0.2 and require explicit rejection. A passing rejection test does not mean the material is supported.
table_full_material_gradients|Check fitting gradients through the complete shiny-plus-matte material, not only its energy table.|Compare twelve derivatives through normalized normal, view, light and material fields against refined quadrature in three cases. Check internal finite differences and a deliberately detached-budget negative control. Shared specular algebra isolates the budget error; it is not independent lobe validation.
table_paired_endpoints_match_original_integrator|Ensure sharing integration work does not change the material model.|Evaluate both Fresnel endpoints from the same geometric samples and compare each to the unchanged scalar oracle at horizon, normal and interior configurations. This is equivalence evidence, not independent physical accuracy.
table_paired_generation_counts_actual_work|Keep the denser table inside the explicit generation-work cap.|Check the new default angle count, invalid inputs, one paired call per knot, two endpoint outputs per geometric sample and rejection of oversized work before integration. Sample reuse does not mean every arithmetic operation costs half as much.
table_retains_old_grid_gradient_failure|Prove that the refined table addresses the measured error without erasing it.|Rebuild the old linear 129-angle grid explicitly, require its known cosine-gradient failure, and compare the quadratic 257-angle candidate to the same reference and unchanged limit. The complete expanded sweep is still required.
table_roughness_grid_is_explicit_and_concentrates_sharp_materials|Keep a rejected experiment reproducible without making it the default.|Check explicit sharp-dense knot placement, fine/coarse nesting, retained uniform defaults and rejection of unknown selectors. The sharp-dense linear trial failed physical checks; correct construction does not imply accurate physics.
table_quadratic_reproduces_known_values_gradients_and_mean|Test interpolation against a problem whose exact answer is known.|Sample a known polynomial, then compare interpolated values, all query gradients and its exact cosine-weighted mean to analytic answers within 1e-12. This isolates the interpolation code from numerical BRDF integration.
table_quadratic_dense_range_screen|Look for unphysical interpolated energy between stored knots.|Screen both Fresnel endpoints on a 65 by 513 roughness/cosine grid, recording finite directional and mean extrema. Quadratic weights can be negative: passing this finite screen is not a proof of shape preservation or energy bounds everywhere.
''')


_add('python/test_material_inputs',
     'Phase-0 field/quadrature recipes and local installed CPU texture/interpolation evidence, not complete material, scene, all-domain energy, CUDA or compiler/export acceptance.', '''
material_fields_expand_only_declared_axes|Prevent point, scene and colour axes from being confused.|Check float32/64 classic and principled schemas, explicit uniform grey parameters, per-point one-channel fields, and output ranks.
material_fields_preserve_shared_gradients_and_noncontiguous_inputs|Keep a shared material learnable across points and scenes.|Expand strided layouts, differentiate every field, and compare exact accumulation counts without input mutation.
material_fields_reject_invalid_or_ambiguous_parameters|Reject wrong material settings before shading instead of guessing their meaning.|Try unknown models, wrong-model or missing fields, invalid values, incompatible batches and ambiguous grey point arrays, including empty calls.
environment_radiance_is_not_cosine_convolved_irradiance|Keep raw incoming SH lighting separate from Lambert's integration shortcut.|Evaluate a constant plus linear-z environment at three directions with known RGB values, including a retained negative channel.
environment_integral_uses_solid_angle_weights_and_cosine_once|Get the units and geometric angle factor right.|Sum three known weighted directions with a Lambert BRDF; compare an independent formula and require doubled solid-angle weights to double the result.
environment_interior_gradients_and_shared_batches|Keep normals, directions, light, response and quadrature weights differentiable.|Run gradcheck for every input and SH coefficients, then verify shared lighting gradients add across two independent scenes.
environment_horizon_masks_have_zero_gradients|Make horizon and behind-surface branches explicit and NaN-free.|Use tangent and back-facing light samples; require zero radiance and zero first derivatives to every input.
environment_empty_repeat_and_validation_contracts|Handle empty work predictably and reject invalid integration data.|Check deterministic repeats, empty points/samples, wrong shapes/devices/dtypes, negative weights/BRDFs, zero vectors, nonfinite input and overflow.
full_coupled_environment_matches_analytic_nonconstant_light|Catch applying Lambert-only environment factors to the complete material.|Integrate the exact roughness-one unit-Fresnel coupled model under 1+0.5*z light and compare independently integrated analytic moments; retain the incorrect shortcut as a negative comparison.
native_vertex_and_constant_texture_material_fields_agree|Connect existing native geometry/texture inputs to the same material contract.|Interpolate constant face-corner values and sample separate 3/1/1-channel maps; compare fields and require gradients to native source inputs.
native_texture_to_material_field_gradients|Ensure fitting can update both texture pixels and UV coordinates.|Gradcheck through native level-zero pyramid sampling into albedo, roughness and reflectance fields on a nonconstant texture away from texel boundaries.
material_environment_demo_and_lesson_are_finite_and_explicit|Make the environment integration distinction inspectable and understandable.|Check 33 samples, four finite response curves, a visible difference from the wrong shortcut, and textbook statements about level zero and solid-angle units.
''')


_add('python/test_material_budget',
     'Scalar float64 CPU quadrature at stated interior fixtures only; not production material, all-domain accuracy, endpoint/world-normal derivatives, CUDA, export or speed acceptance.', '''
budget_has_independent_analytic_value_and_view_derivative|Check reflected energy and view-angle sensitivity against exact mathematics.|At roughness one and unit Fresnel use E(c)=1-c log((1+c)/c) and its analytic derivative at three held-out view cosines.
average_budget_has_independent_analytic_integral|Check the average energy budget without trusting its own numerical sum.|Compare roughness-one unit-Fresnel cosine-weighted average to the exact 4(1-log(2))/3.
budget_held_out_values_and_gradients_against_independent_oracles|Prevent self-consistent but physically inaccurate derivatives from passing.|At three new parameter combinations compare Gauss quadrature to visible-normal integration and compare autodiff to refined direct-grid central differences for view, roughness and Fresnel reflectance.
schlick_budget_is_affine_in_f0_without_detaching|Keep Fresnel and roughness connected to the optimizer.|Compare directional and average budgets to exact interpolation between F0 zero and one; require the corresponding derivative and a nonzero roughness gradient.
coupled_matte_gradients_include_all_three_computed_budgets|Make the matte part respond when the shiny part changes.|Recompute both directional budgets and their average inside gradcheck; verify gradients through all five inputs and reciprocity after swapping view and light.
budget_quadrature_refines_average_values_and_gradients|Check that increasing integration precision stabilizes the average and its derivatives.|Compare 32 versus 64 view nodes with 128 by 256 light samples, at a held-out roughness and F0, without relaxing the 1e-5 limit.
budget_rejects_invalid_domain_type_device_and_work_size|Keep the reference scope and computational cost explicit.|Reject invalid values, endpoints outside this reference, wrong shapes/dtypes/devices, underflow and a request exceeding 2097152 light samples.
budget_repeats_without_mutation_or_retained_parameter_cache|Avoid stale values or freed-autograd-graph reuse between calculations.|Run backward twice on fresh reference calls, checking exact repeat output, unchanged inputs and finite gradients.
budget_demo_is_finite_and_explains_energy_and_gradient_limits|Make energy accounting inspectable without mistaking a constructed identity for proof.|Check 33 finite energy/gradient samples, the explicitly labelled white-matte complement, and textbook links and explanation.
''')


_add('python/test_shade_contract',
     'Phase-0 independent recipes, not public shader/material, hardware parity, compile/export or full-domain gradient acceptance.', '''
lobe_closed_form_values_and_dtype|Fix the exact meaning of each elementary shading term.|Compare float32/64 results to explicit Lambert, Blinn, roughness-one GGX/Smith and Schlick values.
blinn_half_vector_is_not_reflected_vector_phong|Prevent two similarly named highlight models from being confused.|Use a 60-degree light and normal-facing view: the half-vector angle is 30 degrees, not 60.
blinn_approximate_scaling_is_not_energy_conservation|Keep an appearance approximation from acquiring an unsupported energy claim.|Numerically integrate exponent-two Blinn at normal view and compare to the analytic overreflection 25/24.
ggx_projected_distribution_integrates_to_one|Normalize the microfacet distribution by projected area, not radiance.|Integrate D times surface cosine with 65536 midpoint samples at three roughness settings.
microfacet_product_matches_independent_furnace|Check the full D-times-visibility-times-Fresnel product without duplicating its implementation.|Integrate the roughness-one unit-Fresnel case and compare to the analytic 1-log(2).
lobe_interior_gradients|Make all five functions usable in later optimization.|Compare autodiff for each cosine and parameter against central finite differences in the smooth interior.
masked_boundaries_have_finite_zero_gradients|Keep horizon and zero-base exponent conventions explicit and NaN-free.|Differentiate Blinn zero-cosine corners and Smith horizon masks, requiring exactly zero finite gradients.
schlick_endpoint_derivatives_are_not_clamped|Preserve the polynomial's endpoint derivatives for future fitting.|Compare cosine and shared RGB gradients at zero and one against exact derivatives.
lobe_batches_and_shared_gradients_do_not_mix|Prevent one scene's shading parameters from leaking into another.|Compare batched versus independent calls and exact shared-parameter gradient accumulation.
empty_strided_inputs_are_deterministic_and_unchanged|Support tensor views and empty point sets without hidden copies or mutation.|Check repeated strided results, source preservation, empty output shapes and invalid scalar validation even when empty.
lobe_invalid_domains_are_explicit|Reject invalid physical inputs instead of clipping them silently.|Pass out-of-range and nonfinite cosines and roughness values; require explicit ValueError.
lobe_rejects_ambiguous_shapes_types_and_devices|Keep scalar, point and scene sharing unambiguous.|Reject wrong ranks, point counts, additional batches, Python numbers, dtype/device mismatches and invalid albedo/exponents.
small_roughness_is_not_silently_floored|Avoid changing a sharp material to hide numerical trouble.|Check the exact sharp normal-incidence GGX value and reject an underflowed positive denominator parameter.
lobe_demo_and_textbook_explain_model_boundaries|Make the new mathematics visible and understandable.|Check the bounded four-sphere illustration and textbook discussion of half-vectors, energy and Smith visibility.
''')


_add('python/test_cinematic_demo',
     'Finite synthetic presentation fixtures, not public subdivision/material acceptance, correspondence-free inverse rendering or a sustained performance claim.', '''
showcase_refinement_preserves_valid_triangles_and_increases_detail|Improve display mesh detail without replacing the attributed source fixture.|Refine a known triangle and check finite coordinates, four child triangles and valid indices.
demo_highlight_is_reciprocal_and_has_conservative_energy_bound|Keep the demonstration highlight inside an explicit energy budget.|Integrate the shared Phong recipe over 8192 deterministic hemisphere samples at three view angles and check reciprocity.
cinematic_camera_fit_changes_pose_and_focal_and_converges|Recover a rendered shot with actual optimization, not invented camera motion.|Fit 96 known 3D landmarks, require position/orientation and focal changes, and verify small final reprojection error.
png_presentation_assets_round_trip_without_image_dependencies|Save high-resolution visuals without external codecs or unbounded histories.|Decode PNG chunks with stdlib zlib, verify CRCs and pixels, deduplicate assets and enforce a byte cap.
presentation_route_is_confined_and_missing_assets_are_explicit|Keep presentation URLs from exposing arbitrary server files.|Read one content-addressed PNG, then refuse traversal and missing digests with HTTP 404.
cinematic_composition_has_finite_distinct_lighting|Require the showcase to respond to lighting and avoid the observed behind-camera stage failure.|Compare deterministic images under changed lights and check the stage remains in front of both endpoint cameras.
''')

_add('python/test_demo_showcase',
     'Finite local demo and archival fixtures, not public shader/animation acceptance, exhaustive concurrency or sustained performance evidence.', '''
showcase_provenance_obeys_dashboard_contract|Keep demonstration source identity compatible with the dashboard protocol.|Validate revision, dirty state and a source-artifact SHA-256 using the installed dashboard validator.
invalid_demo_provenance_returns_a_json_client_error|Make malformed metadata fail clearly without abruptly disconnecting the caller.|Send an unsupported provenance field and require an HTTP 400 JSON explanation.
archived_demo_can_be_reopened_after_idle_capacity|Keep saved demonstrations viewable after more pages have been visited than can fit in memory.|Open three completed executions with a two-tree limit, revisit the first, and check saved files remain.
archive_capacity_does_not_discard_active_or_unsaved_state|Prevent archive browsing from discarding a live session or unsaved queue frame.|Fill a one-tree server with protected state and require a second execution to be refused.
showcase_scene_uses_offline_meshes_and_valid_topology|Build a recognizable room without adding asset downloads or invalid mesh indices.|Inspect bundled scene sizes, tensor types, finite positions and face-index bounds.
camera_paths_are_finite_and_distinct|Make each named camera demonstration actually change the intended camera state.|Compare finite endpoint transforms and projections for orbit, pan, dolly, zoom and dolly-zoom.
dolly_zoom_keeps_reference_plane_scale_not_background|Explain the dolly-zoom illusion without claiming an entire deep object keeps its shape.|Project foreground-plane and background points at both endpoints; only the foreground matches.
showcase_render_changes_with_light_and_has_depth_normals|Ensure the studio images really respond to light and expose inspectable geometry buffers.|Render a small scene repeatedly, compare changed lights, exact repeat output, depth validity and normal-image shape.
bunny_fit_has_recognizable_subject_and_bounded_convergence|Make the fitting demonstration show an actual object and its optimization progress.|Fit known bunny landmarks for 300 steps and verify 61 distinct-endpoint saved frames and small final pixel error.
depth_reconstruction_reprojects_only_visible_samples|Verify the depth point cloud is reconstructed from the displayed camera, not invented geometry.|Unproject a bounded visible pixel sample and require its reprojection to return the original pixel centers.
ball_energy_breakdown_sums_to_the_reported_mechanical_energy|Explain where the illustrative ball's reported mechanical energy is stored.|Check gravitational, kinetic and spring terms add to the recorded energy and that the energy difference closes the initial balance; this is not independently measured heat.
classic_inspection_uses_native_face_measurements|Make triangle inspection display actual gffx geometry results.|Compute native face areas/normals and vertex-normal colors for the bundled teapot and check shape and area bounds.
depth_demo_publishes_points_through_shared_mesh_panel|Prevent a valid point cloud from being rejected by the testing page catalog.|Publish a small real depth scene and require its four artifacts to bind the point cloud to the shared mesh/points viewer.
''')


_add('packaging/test_ci_contract',
     'A pass confirms these local configuration checks, not a successful GitHub run, a secure third-party tool, or a working graphics operation.', """
action_pins_are_exact_and_immutable|Keep our build tools locked to the revisions we approved, so they cannot silently change underneath us.|Compare five saved GitHub action names, version labels and exact commit IDs with an explicit approved list.
pull_request_matrix_is_the_seven_adopted_lanes|Give proposed changes the agreed quick set of compatibility checks.|Compare the generated seven job names and their Python/PyTorch combinations with the adopted list.
nightly_matrix_is_36_blocking_lanes_with_iso_week_rotation|Cover a broader set of platforms overnight without testing every intermediate version every night.|Generate both odd-week and even-week schedules; check 36 unique required jobs, all platforms, and the alternating PyTorch versions.
release_candidate_matrix_builds_four_and_tests_eighty|Check that release preparation plans to test the same built packages across our compatibility grid.|Check four platform builds, one source archive, and 80 distinct platform/Python/PyTorch installation combinations.
workflows_are_exactly_the_adopted_cadences_and_validate|Keep the automation files aligned with the agreed PR, nightly, release-candidate and manual GPU jobs.|Run the local workflow validator and compare the workflow filenames, including absence of the obsolete wheel workflow.
cuda_lane_cannot_be_started_by_a_fork_or_by_a_schedule|Protect our dedicated GPU machine from untrusted pull-request triggers and unattended scheduled jobs.|Inspect the workflow text for manual dispatch, the designated runner/environment, and absence of PR or scheduled triggers. This is not a live permissions audit.
cuda_lane_enables_the_device_build_and_proves_it_reached_a_device|Avoid calling a GPU job successful when it only ran CPU tests.|Inspect GPU build flags, sanitizer and device-result verification wiring, plus the required device-test names and minimum inventory.
cuda_lane_verifier_rejects_a_pass_that_never_reached_a_device|Make the result checker reject a misleading green report with no actual device tests.|Feed it synthetic test-result inventories, accepting the complete case and rejecting missing device fixtures or too few tests.
workflows_are_least_privilege_pinned_and_scope_truthful|Keep automation permissions narrow and avoid unsupported release or compilation claims.|Search workflow text for required verification commands and read-only settings, and reject a list of unsafe or misleading patterns.
rc_installs_downloaded_wheels_without_rebuilding_source|Test what users would install, rather than accidentally testing a freshly rebuilt replacement.|Inspect the release-candidate consumer job for artifact download, install/uninstall checks, and absence of a source wheel build.
ci_smoke_exposes_installed_and_uninstalled_modes|Keep quick installation checks available to automation.|Inspect the smoke-test script for both command modes and the expected capability and adapter checks.
""")
_add('packaging/test_foundation_verifier',
     'A pass covers the verifier and supplied fixtures. It is not a new full package-foundation run or proof of every installation environment.', """
verifier_declares_every_step_11_gate|Prevent a foundation check from quietly disappearing.|Compare the verifier gate names with the complete adopted eleven-gate list.
ctest_junit_parser_records_the_completed_test_count|Count native test results from machine-readable evidence rather than a green exit code alone.|Parse a sample CTest XML result and require the expected completed-test count.
ctest_junit_parser_rejects_empty_or_incomplete_results|Reject native reports that did not actually complete the required checks.|Feed empty or incomplete XML fixtures to the result parser and require rejection.
ctest_junit_parser_rejects_malformed_or_internally_inconsistent_results|Prevent broken or contradictory native reports from becoming green evidence.|Try malformed XML and inconsistent result-count fixtures and require the parser to reject them.
native_contracts_request_machine_readable_ctest_results|Ensure native verification collects an auditable result count.|Intercept the verifier's commands, supply a sample result, and check the XML-output argument and parsed count.
source_prerequisite_cuda_diagnostic_matches_the_host_contract|Explain missing build prerequisites using the correct platform-specific diagnostic.|Intercept prerequisite commands and compare expected failures and CUDA error text with the simulated host contract.
current_source_inventory_is_clean_and_complete|Keep required source files in the package while excluding disallowed material.|Run the source-hygiene scan on the current inventory and check required paths are present.
hygiene_scan_detects_generated_prototype_and_secret_inputs_without_echoing_secrets|Catch accidental packaging of obsolete code, generated files or secrets without printing secret contents.|Build temporary examples and check their finding categories and absence of the synthetic secret in diagnostics.
base_dependency_scan_rejects_an_undeclared_import|Protect the base package's dependency-light promise.|Scan the real package, then a temporary source fixture with an undeclared import, and require the precise finding.
wheel_inspection_accepts_the_base_contract_and_rejects_runtime_leaks|Stop incorrect binary packages before users install them.|Inspect synthetic good and bad wheel archives for the expected ABI tag, dependency metadata and prohibited bundled content.
sdist_inspection_accepts_source_and_rejects_generated_output|Keep source distributions limited to the intended source contents.|Inspect synthetic source archives, accepting the valid inventory and rejecting generated output.
ignore_policy_covers_local_credentials_and_keys|Reduce the chance of accidentally adding local credentials to Git.|Check that the ignore file contains the required credential and private-key patterns. This does not scan every possible secret.
""")
_add('packaging/test_metadata_contract',
     'A pass means the checked metadata and wording match the adopted contract. It does not independently demonstrate every advertised platform or operation.', """
base_distribution_metadata_is_dependency_free_and_python_310_plus|Declare the correct package identity and minimum Python version without mandatory runtime dependencies.|Read project metadata and compare the package name, development version, Python floor, license, description and empty dependency list.
metadata_declares_supported_python_identity_without_claiming_operations|Avoid presenting an unfinished package as production-stable.|Check classifiers, Python version labels and repository/documentation links against the expected metadata.
development_dependency_groups_are_explicit_and_do_not_leak_to_runtime|Keep build and test tools optional for ordinary users.|Compare development dependency groups and tool pins, and verify that none becomes a mandatory base-package dependency.
readme_states_the_current_product_and_non_support_boundaries|Tell readers what the current repository offers and what it does not yet support.|Check the README for the required current-product and limitation statements.
install_build_dependency_and_support_documents_are_truthful|Keep installation and support instructions explicit about prototype releases and limited evidence.|Check the documentation for required build switches, dependency statements and support caveats.
documented_dependency_categories_match_declared_groups|Make the dependency guide cover every configured optional tool group.|Compare the declared group names with the categories mentioned in the dependency document.
merged_documentation_has_no_branch_or_completed_step_staleness|Remove wording that becomes misleading after a merge or completed packaging work.|Search the documentation for required current statements and reject known stale branch and unfinished-step claims.
""")
_add('packaging/test_wheel_repair_contract',
     'A pass checks repair configuration, not a repaired wheel on every host. PyTorch stays an external optional framework rather than being bundled into gffx.', """
each_platform_has_an_exact_runtime_externalization_policy|Use the agreed Windows, Linux and macOS rules when making portable wheels.|Compare each platform's configured repair command with its exact expected command.
repair_policy_is_narrow_and_never_vendors_pytorch|Exclude only the intended external framework libraries without hiding missing gffx libraries.|Parse repair-command exclusions; reject wildcards, blanket missing-dependency suppression, or excluding gffx_core.
""")
_add('python/test_base_import',
     'A pass covers import and capability-report behavior in this environment. Importing successfully is not proof that every backend or graphics operation works.', """
source_fallback_version_matches_pyproject|Show the same development version whether using source fallback metadata or package metadata.|Compare the source fallback version with the version declared in pyproject.toml.
version_is_a_nonempty_string|Make the package version usable by callers.|Check that gffx exposes a nonempty text version.
base_import_pulls_in_no_forbidden_module|Keep a plain import lightweight and independent of optional frameworks.|Import gffx in a fresh Python process and check that the forbidden optional modules were not imported.
base_import_loads_no_native_library|Avoid loading compiled code just because someone imports the base package.|Inspect loaded modules and native-load state immediately after import in a fresh process.
base_import_does_not_import_the_adapter_namespace|Delay framework integration until a user requests it.|Import the base package in a subprocess and check that the PyTorch adapter namespace is still absent.
capabilities_reports_state_without_probing_a_gpu|Offer a useful environment summary without waking up the GPU.|Inspect capability fields, pointer width and the explicit no-GPU-probe state.
capabilities_tolerates_an_absent_native_core|Explain an unavailable compiled core instead of crashing during inspection.|Check that core capability fields are either a valid loaded ABI/path or an explicit unavailable state with a diagnostic.
unknown_attribute_raises_attribute_error|Give ordinary Python errors for nonexistent gffx attributes.|Request an unknown attribute and require AttributeError with a package-specific message.
dir_reports_the_public_surface|Make Python's attribute listing show the intended public entry points.|Compare dir(gffx) against the explicit sorted public surface.
cuda_namespace_is_lazy_and_probe_is_explicit|Allow discovery of the CUDA namespace without automatically inspecting hardware.|Access the namespace in a fresh process and verify that optional loading/probing has not happened prematurely.
missing_framework_fails_only_the_adapter_import_with_an_actionable_error|Let the base package work without PyTorch and explain how to enable its adapter.|Simulate missing PyTorch in a subprocess, request the adapter, and inspect the installation guidance in its ImportError.
native_core_loads_only_when_capability_state_is_requested|Load the compiled core on demand rather than at base import time.|Compare extension-loaded state before and after an explicit capability request.
""")
_add('python/test_cuda_capabilities',
     'These are mocked capability-report checks, not GPU driver or device conformance tests. A pass does not mean a GPU was present.', """
full_capability_records_receive_stable_names|Turn low-level GPU capability records into consistent, readable Python fields.|Supply a synthetic full native report and compare its named fields, flags and records.
full_capabilities_reports_an_unavailable_native_core|Explain why hardware probing is unavailable when the core cannot load.|Simulate a missing native core and check the returned unavailable report.
public_cuda_namespace_delegates_only_on_call|Keep the public GPU entry point lazy.|Replace the underlying probe with a recording stub and verify delegation occurs only on explicit invocation.
""")
_add('python/test_package_under_test',
     'A pass establishes which local package the suite is testing, not correctness of its operations. An explicit source-tree override changes the intended selection.', """
the_suite_imports_the_installed_distribution_when_one_exists|Avoid accidentally testing source files instead of the built package users would install.|Compare the selected gffx location with the installed distribution, honoring the explicit source-test override.
the_selected_package_carries_its_compiled_core|Ensure the chosen test package includes its native extension.|Inspect the selected package for the compiled core required by this verification environment.
""")
_add('pytorch/test_adapter_loading',
     'A pass checks the adapter loading boundary or its build source in this environment. It does not certify all PyTorch releases or graphics operations.', """
import_gffx_exposes_a_lazy_torch_namespace_without_loading_it|Keep the PyTorch adapter available on demand without loading it at base import.|Inspect the public namespace and subprocess module state before an explicit adapter request.
below_floor_pytorch_fails_only_adapter_access_with_detected_version|Tell users clearly when their PyTorch version is too old.|Simulate a below-floor framework and check that base import works but adapter access reports the detected unsupported version.
registration_source_uses_only_the_libtorch_stable_api|Keep the registration layer on the intended stable PyTorch binary interface.|Inspect adapter source for the required stable API and prohibited unstable registration patterns.
adapter_build_pins_the_python_and_libtorch_abi_floors|Build the adapter against the intended minimum binary interfaces.|Inspect build definitions for the adopted Python and LibTorch ABI floors.
supported_pytorch_loads_the_private_registration_probe|Prove that requesting the adapter registers its internal operation with PyTorch.|Load the adapter under supported PyTorch and invoke or inspect the private foundation registration probe.
""")
_add('pytorch/test_face_geometry',
     'A pass covers the tested mesh fixtures and adapter behavior only. C reference parity checks agreement with that implementation, not an independent proof for every mesh. Skips are not passes.', """
ta01_forward_parity|Compute each triangle's normal direction, area and validity without changing the C core's answers.|Compare PyTorch outputs bit-for-bit with committed C reference outputs for a tetrahedron and a mesh containing a degenerate face. The six child pages separate these comparisons.
ta02_vertex_dtypes|Accept the promised coordinate precisions and reject unsupported narrow formats.|Run with float32 and float64 vertices, check output types, and require refusal of narrower input types.
ta03_non_contiguous_is_refused|Avoid a hidden memory copy when mesh coordinates use a strided layout.|Pass a non-contiguous tensor view and require an explicit error.
ta04_int64_faces_are_refused|Make the required triangle-index format clear.|Pass 64-bit face indices and require an error directing the user to int32 indices.
ta05_device_policy|Do not silently move mesh data between CPU and GPU.|Pass vertices and faces on different devices and require a same-device error; the hardware-dependent case may skip.
ta06_backward_parity|Preserve the C core's answers when propagating gradients back to mesh vertices.|Supply saved output gradients and compare the resulting vertex gradients bit-for-bit with the committed C backward reference.
ta07_gradcheck|Check that the area gradient matches small measured changes in vertex positions.|Use PyTorch numerical differentiation to compare the registered backward formula on the selected fixture.
ta08_absent_cotangents|Handle callers who differentiate only some outputs.|Exercise missing output-gradient inputs, require finite gradients where valid, and reject a call with no gradient inputs.
ta09_error_mapping|Translate native failures into useful Python exceptions.|Trigger the reachable native error categories and check their exception types and nonempty messages.
ta10_input_immutability|Keep the caller's vertices and face indices unchanged during calculations.|Save copies, run forward and backward calls, and compare inputs afterward.
ta11_determinism|Return exactly the same answers for repeated identical calls.|Run the same mesh calculation repeatedly and compare output tensors bit-for-bit.
ta12_out_surface|Support inference using caller-owned output buffers without accidentally building a gradient graph.|Compare preallocated outputs with ordinary outputs and reject inputs that require gradients on this inference-only entry point.
ta13_single_face_gradient|Prevent a one-triangle mesh from failing during backpropagation.|Differentiate a one-face mesh, including the size-one gradient layout that previously caused a stride error, and require finite gradients.
""")
_add('pytorch/test_all_operations',
     'A pass covers this small adapter fixture, not exhaustive kernel correctness, speed, or all devices. CPU/GPU exact equality is claimed only for the explicitly parameterized comparison group.', """
tb01_vertex_normals|Calculate usable per-vertex directions and wire their gradients to vertices.|On a tetrahedron, check shape, precision, unit length, differing weighting modes, finite gradients and invalid-weighting rejection.
tb02_gather_faces|Gather the right three vertices for every triangle.|Compare the result with direct PyTorch indexing and check each tetrahedron vertex receives the expected accumulated gradient.
tb03_transform_points|Apply a transformation matrix to points with the correct argument layout.|Compare with an explicit PyTorch matrix product and check that gradients reach the input points.
tb04_perspective_divide|Divide projected coordinates safely, including points whose divisor is zero.|Check a known valid result, a zero-filled invalid row, validity flags and finite gradients.
tb05_build_edge_topology|List mesh edges and their incident triangles correctly.|Use a tetrahedron with six known edges; check endpoint order, two incident faces per edge and unused-row sentinels.
tb06_knn|Find the nearest reference points to each query.|Compare distances and indices with a brute-force PyTorch calculation and check output types and query gradients.
tb07_closest_point_on_mesh|Find a point's closest location on a triangle surface.|Use a point below a known face; check its projected location, squared distance, triangle weights and analytic distance gradient.
tb08_sample_surface|Generate repeatable mesh-surface samples with explicit random state.|Reuse the same key and counter, then advance the counter; check repeatability, changed samples, valid faces, triangle weights and finite gradients.
tb09_rasterize|Turn triangles into correctly shaped pixel-fragment outputs.|Render a small fixture, check covered pixels and output types, and check the intended gradient behavior.
tb10_interpolate|Carry per-vertex attributes into covered pixels.|Interpolate a constant attribute across a rendered mesh and check that covered pixels keep that value and gradients remain finite.
tb11_ply_reader|Load a PLY mesh into tensors that geometry operations can use directly.|Read the same fixture from bytes and a file, check coordinates/index types, and pass the result into face geometry.
tb12_validate|Report mesh defects without stopping at the first bad item.|Survey clean and deliberately invalid meshes; check range findings, optional finite-coordinate checks and degenerate-face counts.
tb13_streaming_surface|Make caller-preallocated inference operations match their ordinary counterparts.|Compare the streaming entry points with functional outputs and require rejection of gradient-tracked inputs. This is not a latency benchmark.
tb14_cuda_conformance|Keep CPU and GPU answers identical for the selected order-independent operation.|Run the parameterized fixture on both devices and require matching shape, dtype, GPU placement and bit-for-bit values after copying back for comparison.
tb14_unimplemented_operation_is_refused|Avoid silently falling back to CPU for a missing GPU operation.|Request an operation without a CUDA kernel and require an explicit refusal instead of hidden device transfers.
tb14_mixed_devices_are_refused|Catch mixed-device arguments before a GPU kernel is called.|Pass mismatched CPU/GPU tensors and require an informative device-mismatch error.
""")
_add('python/test_camera',
     'These fixtures describe the intended camera contract. Expected failures mean the feature is not yet accepted; skips mean it was not exercised. A real pass covers only the stated fixture and tolerance.', """
cam10_world_to_clip_is_bit_identical_to_the_hand_assembled_matrix|Compose camera position and projection in the correct order.|Build the projection and view matrices by hand and require exact equality with the camera matrices for the selected precision.
cam11_project_is_bit_identical_to_the_manual_chain|Put a 3D point at the expected image location.|Compare camera projection with the explicit transform, perspective divide and pixel conversion chain, including depth and validity.
cam12_ndc_depth_matches_the_rasterizer_at_a_vertex_on_a_pixel_centre|Use the same depth convention in camera projection and rendering.|Place a vertex at a pixel centre and compare its projected depth with the rasterizer's depth within the fixture tolerance.
cam13_view_depth_and_ndc_depth_convert_through_the_projection_row|Convert camera-space depth to normalized rendering depth and back.|Round-trip known depths and check the near/far boundary values against the projection convention.
cam14_unproject_recovers_interpolated_world_positions_from_a_rendered_depth_buffer|Recover 3D surface positions from a rendered depth image.|Compare reconstructed covered pixels with interpolated world positions; check validity and zero-filled background.
cam15_unproject_points_inverts_project|Recover original 3D points from their image coordinates and depth.|Project known points and unproject them through the supported depth paths, comparing within the fixture tolerance.
cam16_stack_builds_a_rig_that_equals_the_single_camera_runs|Make a batch of cameras behave like the same cameras used individually.|Stack two cameras and compare each batch result exactly with its separate projection.
cam17_gradcheck_reaches_every_intrinsic_and_the_pose_of_each_rig_camera|Allow optimization of camera lens parameters and positions.|Compare analytic and numerical gradients for focal lengths, image centres and both camera poses, and check nonzero gradient flow.
cam17_cuda_gradcheck_on_device|Check camera differentiation while the computation runs on a GPU.|Gradcheck all four independent intrinsic tensors and the full extrinsic matrix on CUDA in float64 when the required device is available.
cam18_invalid_construction_is_rejected|Reject camera settings that violate the public contract.|Try the parameterized invalid-construction case and require the documented exception.
cam19_unproject_of_an_infinite_background_is_zero_valid_false_and_free_of_nan|Handle empty depth-image background without poisoning coordinates or gradients.|Unproject infinite background depths; require invalid flags, zero background coordinates and gradients, and no NaNs.
cam20_the_facade_computes_nothing|Keep the convenient camera API consistent with the framework-specific API.|Run equivalent calls through gffx.camera and gffx.torch.camera and require exact output equality.
camera_plain_numbers_and_inert_hooks|Construct a basic camera without creating tensors by hand.|Project an optical-axis point and check image coordinates, batch size and inactive future camera features.
camera_additional_construction_errors|Reject invalid image sizes, focal lengths, poses and tensor types before computing.|Try each explicitly invalid constructor input and require a ValueError or TypeError.
camera_packed_rig_matches_single_cameras|Let different cameras have different numbers of input points without padding.|Project a packed two-camera batch and compare every output exactly with separate camera calls; reject missing or invalid offsets.
camera_empty_dense_and_packed_batches|Make an empty set of points an ordinary result, not a crash.|Project empty dense and packed camera batches and check that output shapes preserve the empty dimensions.
camera_unprojection_mask_and_invalid_kind|Keep missing depth samples out of coordinates and gradients.|Unproject a masked depth grid, check zero gradients for the rejected sample, and reject invalid depth labels, shapes and masks.
camera_projection_dtype_cast_and_input_immutability|Preserve the caller's point precision and original data.|Project float32 points through float64 camera parameters; check output precision and device and compare saved input copies.
camera_mixed_devices_are_rejected|Avoid secretly copying camera data from CPU to GPU.|Give a CPU camera GPU points and require a same-device error; this check skips if no CUDA device is available.
camera_stack_rejects_incompatible_clipping_and_empty_list|Keep a camera rig's shared settings unambiguous.|Require errors when stacking cameras with incompatible clipping settings or when no cameras were supplied.
camera_front_filter_preserves_projection_validity_not_visibility|Offer an optional behind-camera rejection without confusing mathematical projection with what is visible.|Compare default and front-only masks for front, behind, camera-plane, off-screen and beyond-far points on the selected device; check unchanged accepted outputs and zero rejected gradients.
""")
_add('python/test_testing_dashboard',
     'These tests cover local presentation routing and a finite demo producer, not physical-phone acceptance or graphics correctness.', """
demo_pages_do_not_borrow_another_pages_artifacts|Keep each demonstration page attached to its own visual evidence.|Publish distinct panel and camera demos, check page selection and reject switching an execution to another page or a test page.
camera_demonstration_is_finite_and_reports_its_own_page|Provide bounded, inspectable camera examples without a continuous stream.|Publish labelled landmarks, depth reconstruction and two-view illustrations to a temporary server; require ten artifacts, labelled image dimensions and a finished non-test execution on the camera demo page only.
catalog_cache_invalidates_new_evidence_and_keeps_live_staleness|Can faster navigation conceal a newer failure or stale worker?|Require repeated reads to reuse only the catalog projection, invalidate after every evidence change and pin, retain newer partial and failed runs, refresh stale status by time, and prevent caller mutation of cached data.
""")

for _test,_name,_purpose,_method in (
    ('cam10_world_to_clip_is_bit_identical_to_the_hand_assembled_matrix','matrix convention','Keep one world-to-image matrix convention.','Compare projection, inverse pose and combined matrices exactly with the manual reference; display expected and actual entries.'),
    ('cam16_stack_builds_a_rig_that_equals_the_single_camera_runs','rig equals separate cameras','Give each camera in a batch exactly its own projection.','Compare both batched pixel arrays with their separate camera calls bit-for-bit and display a small sample and maximum error.'),
    ('cam19_unproject_of_an_infinite_background_is_zero_valid_false_and_free_of_nan','background outputs and gradients','Keep empty depth-image pixels harmless during reconstruction and optimization.','Unproject an 8 by 8 grid with nine finite samples; require zero background coordinates and gradients and display mask counts and maximum background gradient.'),
    ('cam15_unproject_points_inverts_project','metric depth round trip','Recover points from pixel positions and ordinary camera depth.','Project and reconstruct 256 points, record the maximum world-coordinate error and require at most 1e-9 absolute error.'),
    ('cam15_unproject_points_inverts_project','NDC depth round trip','Recover points using normalized renderer depth.','Project and reconstruct the same 256 points via NDC depth, record the maximum world-coordinate error and require at most 1e-9 absolute error.'),
    ('camera_front_filter_preserves_projection_validity_not_visibility','mathematical validity','Keep behind-camera projections valid when their division is mathematically defined.','Compare the default mask with the existing manual transform/divide chain for five named point categories.'),
    ('camera_front_filter_preserves_projection_validity_not_visibility','optional front filter','Reject behind-camera points only when explicitly requested.','Require the front-only mask and zero rejected outputs while keeping off-screen and beyond-far front points valid.'),
    ('camera_front_filter_preserves_projection_validity_not_visibility','rejected point gradients','Prevent filtered-out finite points from affecting optimization.','Backpropagate through all returned numeric fields; display gradients and require finite, exactly zero gradients on rejected points.'),
):
    CHECK_GUIDES[('tests/python/test_camera.py::test_'+_test,_name)]=dict(purpose=_purpose,method=_method,
        limits='This named fixture is not occlusion, raster clipping, compilation/export acceptance or a general proof for all scenes; hardware claims apply only to the recorded device.')
_add('python/test_dashboard',
     'A pass covers the local server/client fixture. It is not physical-phone acceptance, a sustained resource benchmark or proof of visual rendering conformance; skipped display checks remain unmeasured.', """
db01_set_reaches_a_page_as_snapshot_then_update|Show a page its initial value and subsequent live changes.|Connect a client and page, send values, and compare the snapshot and update messages.
db02_mesh_survives_binary_framing_bit_for_bit|Transport mesh numbers without changing them.|Send mesh buffers over the dashboard connection and compare received binary data bit-for-bit.
db03_push_past_the_bound_evicts_the_oldest|Keep recent history from growing beyond its configured window.|Push more values than the window allows and check that only the newest bounded history remains.
db04_a_control_set_on_the_page_reaches_the_client_as_a_live_value|Let an interactive page control a connected producer.|Change a control through the page connection and check the producer receives its updated value.
db05_a_run_log_replays_to_the_same_tree|Restore persisted dashboard state after reopening it.|Record a run, replay its log and compare the reconstructed page tree.
db06_layouts_round_trip|Remember how a user arranged dashboard panels.|Save a layout and read it back, checking the serialized layout survives the round trip.
db07_the_server_binds_only_the_tailscale_interface|Avoid accidentally exposing the default server on every network interface.|Exercise address selection/binding and require the intended Tailscale address rather than a wildcard listener.
db08_the_pytest_plugin_posts_outcomes_and_evidence|Send test results and visual evidence to the general dashboard.|Run a small pytest producer and inspect its reported outcomes and evidence panels.
db09_no_address_means_every_call_is_a_no_op|Let instrumentation remain harmless when no dashboard is configured.|Use a disabled connection and exercise its methods without requiring a server or producing work.
db10_a_full_queue_costs_the_sender_under_the_budget|Avoid stalling the producer when dashboard delivery is overloaded.|Fill the sender queue, time the fixture's additional sends and compare with the configured local time budget.
db11_the_viewport_matches_a_gffx_cpu_render_to_a_display_tolerance|Eventually compare the browser mesh display with a CPU-rendered reference.|This acceptance fixture is currently skipped because framebuffer comparison is not implemented. No visual conformance result exists yet.
db12_mixed_kinds_on_one_path_are_rejected_and_the_connection_survives|Reject incompatible panel updates without breaking future valid updates.|Send conflicting panel kinds to one path, require an error, then confirm the connection still works.
""")
_add('python/test_dashboard_checkpoints',
     'A pass covers the configured small fixtures. Accounted payload bytes are not total process RAM, and these tests do not establish sustained phone performance or crash-proof filesystem durability.', """
recent_window_and_sparse_checkpoints_are_independent|Keep a short live buffer while saving history much less often.|Send steps 0 through 12 with a three-frame window and five-step checkpoint interval; compare live frames, saved frames and expired-frame errors.
checkpoint_replay_without_shutdown_tail|Restore only the intended saved checkpoints, not every recent live frame.|Use a two-frame window and three-step snapshots, disable shutdown tails, reopen the log and compare restored frames.
disabled_checkpoints_keep_no_durable_frames|Allow a live-only viewer with no saved queue history.|Set the snapshot interval to zero, send values and check that neither checkpoints nor a shutdown tail are written.
source_throttle_never_skips_due_checkpoint|Reduce ordinary live traffic without losing scheduled snapshots.|Hold a fake clock steady while sending steps under a rate limit; check that due checkpoint steps still get through.
invalid_queue_policy_rejected|Catch invalid buffer sizes, snapshot intervals and update-rate settings early.|Try each parameterized invalid policy and require rejection without accepting ambiguous values such as booleans or NaN.
settings_can_change_without_erasing_checkpoints|Let users adjust the live window and snapshot cadence without deleting old saved history.|Change both settings mid-stream and compare the new recent window with retained old and new checkpoints.
checkpoint_quota_error_is_visible_and_does_not_mutate_live|Report a full history budget without partially applying a failed checkpoint.|Fill the disk quota, reject a due checkpoint and compare unchanged state; verify ordinary unsaved live updates can still proceed.
defaults_and_source_steps_before_copy|Keep sensible defaults and reject invalid or repeated step numbers.|Check the 100-frame/1000-step defaults and try backward, repeated, negative and non-integer steps without changing state.
mesh_checkpoint_self_contained_after_live_eviction|Keep a saved mesh viewable after its live frame has disappeared.|Save a small triangle, evict it from the live window, then retrieve its saved vertices and topology; also verify deletion invalidates access.
pending_checkpoints_are_protected_and_overload_is_visible|Protect a queued snapshot from being replaced silently by ordinary live traffic.|Use a one-slot sender queue with a stalled worker and inspect drop/error reporting when ordinary updates and more checkpoints arrive.
due_checkpoint_evicts_ordinary_pending_update|Prioritize a scheduled saved frame when the sender is full.|Fill the queue with an ordinary update, enqueue a due checkpoint and verify the ordinary update is replaced instead.
recent_window_still_respects_bytes_and_checkpoint_pages_are_bounded|Bound both live payload memory and each history-list response.|Send 130 values under a small byte/window budget and check checkpoint pagination and retrieval of an old saved step.
slow_checkpoint_scan_does_not_hold_producer_lock|Keep history browsing from blocking new producer updates.|Block a simulated checkpoint read in another thread and verify the producer lock remains available.
""")
_add('python/test_dashboard_resources',
     'A pass checks explicit local limits and simulated overload cases. Payload accounting is not total RAM, and short loopback timing is not a sustained mobile or network performance guarantee.', """
set_history_and_queue_are_bounded_and_report_evictions|Keep both saved-state history and live queues within their configured frame counts.|Send ten updates into smaller windows and compare retained values and eviction totals.
byte_budget_evicts_old_values_and_rejects_oversize_without_mutation|Bound payload bytes as well as frame count.|Send growing text values, check byte-budget evictions, then reject one oversized value without altering the accepted state.
total_run_budget_and_path_count_are_bounded|Prevent many small panels from bypassing the overall limits.|Fill a small run-byte budget and panel-count allowance, then require refusal beyond those limits.
log_quota_preserves_existing_bytes|Leave previous saved evidence intact when a log is full.|Attempt an append beyond a tiny log quota and compare existing bytes before and after rejection.
page_delivery_coalesces_and_hidden_page_keeps_no_pending_frames|Send the newest live value instead of building a backlog for a slow or hidden page.|Queue 1000 same-path updates, check that only the latest is pending, then pause delivery and check the pending queue stays empty.
mesh_delta_is_connection_local_and_changed_topology_is_resent|Save bandwidth without leaving a viewer with stale triangle connectivity.|Compare first/full and subsequent vertex-only mesh messages; change topology or reset connection state and require full necessary data again.
resource_limits_validate_positive_integers|Reject nonsensical resource budgets before they are used.|Try zero, negative, fractional and boolean limits and require validation errors.
disk_limit_rejection_does_not_change_live_state|Keep a failed saved-state update from changing what the live viewer considers accepted.|Use a real local server, exceed its disk quota and check the preceding live value remains intact.
sender_queue_has_a_byte_budget|Bound pending producer messages by bytes, not only message count.|Place one 60-byte message in a 100-byte queue, reject another, then check draining releases the accounting.
slow_page_socket_does_not_hold_sender_queue_lock|Let producers enqueue while a viewer socket is slow.|Block a fake socket write and check a separate enqueue can still finish within the fixture timeout.
page_pacing_and_pause_resume_over_wire|Respect viewer frame pacing and pause/resume controls.|Observe local wire delivery at two updates per second, pause, then resume and require the latest value.
full_disk_does_not_partially_append_shutdown_tail|Do not leave a half-written legacy queue tail when disk allowance is exhausted.|Use a tiny disk cap, attempt tail saving and require a warning with unchanged existing log bytes.
negotiated_mesh_delta_and_resume_send_full_frame|Reestablish a complete mesh after a paused viewer resumes.|Negotiate mesh deltas over a local connection, observe vertex-only updates, then require a full frame after resume.
disk_budget_is_shared_across_runs|Prevent multiple runs from each consuming the entire global log budget.|Append through two run logs under one small quota and reject the second write when their combined budget is exhausted.
page_byte_budget_rejects_oversize_and_keeps_latest|Protect a slow viewer's outgoing queue without losing the latest accepted update.|Fill a small page-byte budget, check latest-value retention and reject a single oversized frame with visible drop accounting.
""")
_add('python/test_dashboard_references',
     'A pass covers local identity and metadata preservation. Producer provenance is reported metadata, not cryptographic proof that the producer ran unmodified code.', """
page_identity_survives_save_and_restart|Keep a shared dashboard page link valid after layout edits and server restarts.|Save a layout and restart twice, checking that its run and page identities stay stable.
page_ids_distinguish_runs_and_layouts|Avoid sending a shared link to the wrong page.|Create two runs with three layouts each and require six distinct page identifiers.
run_provenance_is_reported_and_cannot_be_rewritten|Keep the original producer revision attached to saved evidence.|Record revision, dirty state and artifact metadata, reopen the log and reject conflicting replacement provenance.
unknown_provenance_is_not_inferred_from_server|Do not pretend the server's source revision belongs to an unknown producer.|Start a run without provenance and require it to remain explicitly unknown.
invalid_or_conflicting_provenance_does_not_poison_log|Reject bad provenance without corrupting previous evidence.|Send malformed or conflicting metadata and compare the existing log bytes after refusal.
""")
_add('python/test_dashboard_persistence',
     'These fixtures cover legacy push/shutdown-tail compatibility. They do not override the newer live-window plus sparse-checkpoint policy or promise saving after an abrupt crash.', """
queue_tail_restarts_preserve_steps_and_next_push|Resume a legacy saved queue without resetting its step numbers.|Save a queue tail, reopen it and send another push, checking retained steps and the next assigned step.
push_is_unlogged_until_clean_shutdown|Avoid writing each legacy push to disk as it arrives.|Inspect log contents before and after a clean shutdown and verify the tail is saved only at that boundary.
""")
_add('python/test_dashboard_touch',
     'A pass means the automated JavaScript interaction contracts pass. It is not a test performed on an actual iPhone touchscreen or a visual WebGPU conformance result.', """
dashboard_touch_contracts|Keep browser interaction behavior covered by repeatable tests.|Run the shared dashboard's Node-based JavaScript tests and require a successful result.
""")
_add('python/test_testing_dashboard',
     'A pass covers the tester application fixture, not the graphics feature whose results it displays. Synthetic test results and demos are not product acceptance, and Python coverage is not correctness.', """
every_repository_test_has_an_authored_explanation|Keep future test pages understandable instead of presenting unexplained internal IDs.|Inspect real repository test definitions, including class methods but excluding embedded sample scripts, and require a substantial authored guide for each one and its parameter variants.
explanations_are_read_only_guides_not_historical_test_results|Explain old pages without rewriting what their tests actually reported.|Open a pending known test and an unknown external test; check authored-guide provenance, honest missing-guide behavior and unchanged pending status.
named_check_explanations_describe_the_comparison_and_its_limits|Explain each small mesh comparison without turning its description into a pass.|Register all six face-geometry checks and require their bit-exact comparison guide while the results remain pending.
individual_and_named_check_pages_isolate_artifacts_and_never_fallback|Keep each test and named check tied to its own evidence.|Create two tests and a child check, attach artifacts, restart the index, then collect a newer execution; verify ownership and no fallback to an old check result.
named_check_lifecycle_and_quotas_fail_closed|Reject duplicate, out-of-order or excessive named-check reports.|Send invalid lifecycle events and exceed a small check quota; check refusal, stable counts and bounded directory responses.
named_check_fixture_reports_failure_without_swallowing_assertion|Make check reporting preserve ordinary test failures and skips.|Use a temporary recording reporter, execute passing/failing/skipped check scopes, and verify both emitted events and propagated exceptions.
missing_call_report_cannot_finish_green|Prevent a missing test-body result from becoming a successful execution.|Report collection and teardown but omit the call result, finish with exit code zero, and require an incomplete summary.
progress_counts_tests_not_reports_and_keeps_teardown_failure|Count each test once and retain failures that happen during cleanup.|Report setup, call and failing teardown phases and check test totals and final failure precedence.
more_than_128_tests_are_paginated_and_survive_restart|Handle a large test inventory without treating every test as a visual panel.|Collect more than 128 tests, inspect paginated results and reopen the index to verify preservation.
heartbeat_stale_is_not_completion_and_demo_is_not_a_test|Distinguish a disconnected producer and an illustrative demo from a completed test run.|Advance the fixture clock and inspect stale running status; create demo evidence and verify it does not inflate test totals.
coverage_explicit_unknown_and_validated_counts|Show missing coverage honestly and accept only coherent measured counts.|Inspect an unmeasured run, then submit valid and invalid coverage summaries and check validation behavior.
quota_rejection_is_transactional_and_unknown_ids_do_not_fallback|Refuse excess or unknown data without substituting unrelated evidence.|Exceed small index limits and request unknown identifiers, checking transaction rollback and explicit errors.
separate_http_app_reuses_general_assets_and_rejects_cross_origin|Keep the tester separate while reusing shared panels and guarding writes.|Serve the local app, inspect shared asset routes, and require rejection of foreign-origin requests.
offline_reporter_never_raises_even_with_warnings_as_errors|Keep a disconnected dashboard from breaking a producer test.|Send through an unavailable reporter under strict warning handling and require safe failure reporting rather than an exception.
real_pytest_session_reports_progress_outcomes_evidence|Prove the reporting bridge works with an actual pytest subprocess.|Run sample tests and inspect collected counts, outcomes and published visual evidence in the local tester.
testing_frontend_contracts|Keep the tester's navigation, references, themes and display logic under regression tests.|Run its Node-based JavaScript contract suite and require a successful result.
demo_is_finite_persisted_and_uses_shared_visual_data|Provide reusable demonstrations without a never-ending background generator.|Run the finite demo publisher, inspect its saved shared-panel data and verify it remains marked as a demonstration.
finish_never_fabricates_success_or_overwrites_terminal_result|Keep completion reports honest and final.|Try incomplete, contradictory or repeated finish events and check they cannot create or overwrite a successful terminal result incorrectly.
offline_plugin_preserves_test_outcome_under_warnings_as_errors|Keep optional reporting from changing real pytest outcomes when the server is absent.|Run a subprocess with strict warnings and an unreachable dashboard and compare its test outcome.
cli_collects_real_python_coverage_when_tool_is_available|Attach actual Python execution coverage when the optional development tool is installed.|Run the CLI with coverage enabled and inspect its produced coverage report and submitted counts.
coverage_boundary_rejects_malformed_file_rows|Prevent malformed per-file coverage from polluting the coverage page.|Submit inconsistent or invalid coverage-file rows and require validation failure.
explicit_existing_run_refused_before_executing_tests|Avoid overwriting saved evidence by reusing an execution identifier.|Start with an existing run ID and verify the CLI refuses it before launching the requested tests.
catalog_pages_exist_before_execution_and_reject_unknown_pages|Make feature pages permanent even before tests produce results.|Inspect the initial catalog for unmeasured pages and require errors for unknown page names.
page_selects_latest_relevant_execution_not_latest_success|Show recent failed or incomplete evidence rather than an older green result.|Create successive relevant and unrelated executions and check page selection follows the newest relevant collection.
page_pagination_counts_are_local_and_filters_do_not_change_summary|Let users filter a feature's test list without changing its overall reported result.|Paginate and filter a synthetic feature inventory and compare unchanged feature-local summary counts.
demo_page_is_permanent_and_independent_of_test_execution|Keep demonstrations available separately from test progress.|Publish demo evidence and later test runs, then check the demo page still selects its own illustrative data.
page_api_requires_no_run_and_keeps_general_dashboard|Open feature links without asking users to choose an execution first.|Exercise the page API without a run parameter and verify the separate tester and general dashboard routes coexist.
catalog_routes_existing_test_families_without_silent_loss|Group known tests under their intended features and make unknown routing explicit.|Check representative repository test paths against the catalog and verify unmatched cases use the unassigned group.
page_creation_dates_are_explicit_and_independent_of_execution|Sort pages by when they were registered, not when they last ran.|Create/reopen pages across executions and inspect stable creation dates and directory ordering.
""")

for _mesh in ('tetrahedron', 'degenerate_face'):
    for _field, _meaning in (
        ('normals', 'the direction perpendicular to each triangle'),
        ('areas', 'the surface area of each triangle'),
        ('validity', 'which triangle results are marked usable by the face-geometry contract'),
    ):
        CHECK_GUIDES[('tests/pytorch/test_face_geometry.py::test_ta01_forward_parity', _mesh+' / '+_field)] = dict(
            purpose='Check '+_meaning+' for '+('a four-faced tetrahedron.' if _mesh=='tetrahedron' else 'a fixture containing a collapsed (degenerate) triangle.'),
            method='Compare the actual PyTorch '+_field+' tensor bit-for-bit with the committed C reference tensor. The comparison record shows both; normal checks also display the input mesh.',
            limits='A pass means this field matches the C reference for this fixture. It does not independently validate the reference, other fields, gradients, rendering, or all meshes.')


_add('python/test_testing_dashboard',
     'These are importer and evidence-isolation fixtures, not coverage measurements of the fixture repository or GPU kernels.', """
coverage_uses_statement_summary_not_raw_execution_lines|Use the executable-statement denominator reported by Python instrumentation.|Provide extra raw executed-line entries and assert the authoritative summary defines the displayed numerator and denominator.
lcov_layers_preserve_independent_reports_and_unknown_gaps|Keep languages and executions separate instead of presenting a false repository-wide percentage.|Import synthetic JavaScript LCOV with missing lines and branches beside Python evidence; verify coexistence, historical selection and no fallback for an unmeasured native layer.
coverage_inventory_never_calls_uninstrumented_files_zero_coverage|List source files that lack instrumentation without inventing executable-line counts.|Inventory a small repository with Python, JavaScript and CUDA source and reject a path outside its root.
""")


_add('python/test_camera',
     'Smooth synthetic fixtures away from validity boundaries. Passing derivatives do not imply visibility derivatives, real-data identifiability, export support or camera performance guarantees.', """
camera_phase3_unprojection_gradients|Check how reconstructed 3D positions change when camera settings, depth or image coordinates change.|Compare autodiff with central finite differences for all four intrinsics, the pose matrix and depth, plus scattered pixel coordinates, for view and NDC depth on CPU and CUDA.
camera_phase3_stacked_rig_gradients_are_independent|Ensure one optimization can adjust each camera independently.|Backpropagate a combined two-camera loss through Camera.stack, compare each gradient with independent single-camera computations and require zero cross-camera influence for the first camera's loss.
camera_phase2_cuda_project_unproject_preserves_device|Verify camera compositions actually execute with GPU tensors and retain their device.|Compare CPU and CUDA projection/unprojection for float32 and float64 and inspect every returned tensor's device; no new camera CUDA kernel is claimed.
""")
_add('python/test_testing_dashboard',
     'This checks a deterministic, identifiable synthetic intrinsic fit with fixed geometry and pose. It is not real-data calibration or the gffit optimization engine.', """
camera_fitting_demo_has_bounded_history_and_correct_derivative_curve|Make sure the camera fitting demonstration actually improves and remains finite.|Run 300 steps, require 61 retained scalar checkpoints, compare recovered intrinsics to known targets and compare 21 inverse derivatives with central finite differences.
""")
for _test,_checks in {
    'camera_phase3_unprojection_gradients':['dense inverse derivatives','scattered inverse derivatives'],
    'camera_phase3_stacked_rig_gradients_are_independent':['independent rig backward'],
    'camera_phase2_cuda_project_unproject_preserves_device':['inherited CUDA composition'],
}.items():
    _node='tests/python/test_camera.py::test_'+_test
    for _name in _checks:CHECK_GUIDES[(_node,_name)]=dict(GUIDES[_node])


_add('python/test_testing_dashboard',
     'Synthetic demo/data-lifecycle checks only: not animation-operator correctness, rendering conformance or a playback performance guarantee.', """
mesh_animation_demo_is_finite_and_replays_frames|Ensure mesh and animation examples publish a fixed amount of data and survive a server restart.|Publish nine artifacts, inspect both endpoints of a 24-frame clip, reject a nonexistent next frame and restart the server to read retained 2D and 3D frames again.
mesh_animation_geometry_has_valid_topology_and_changes_over_time|Ensure the examples contain connected triangle meshes with valid indices and actual movement.|Check finite vertices, distinct triangle corners, stable topology, changing positions and a strictly planar Z coordinate for the 2D example.
""")


_add('python/test_testing_dashboard',
     'Developer demo fixtures only. No general physics, commercial data permission, browser FPS, mesh validity or differentiability guarantee is inferred.', """
classic_demo_assets_are_offline_bounded_and_attributed|Keep recognisable mesh examples lightweight, offline and traceable.|Read both bundled assets, check finite coordinates and triangle indices, size bounds, source and license metadata, and reject unknown asset names.
elastic_ball_freefall_rebound_energy_and_convergence|Check that the ball illustration follows its stated simple contact model.|Compare pre-contact height with analytic gravity, require rebound and damping, bound compression/energy and compare against a halved integration timestep.
elastic_ball_scene_squashes_without_floor_penetration|Keep the visible squashing ball above its rigid floor with stable triangles.|Inspect all 181 generated frames for fixed topology, bounded mesh size, changing positions and nonnegative vertical coordinates.
classic_demo_publishes_finite_assets_physics_and_restart_history|Keep classic meshes and ball playback finite, separate from test acceptance and recoverable after restart.|Publish two demo pages, inspect saved first/last frames, reject an extra frame, confirm producer completion and reload both clip endpoints after server restart.
""")


_add('python/test_shading_contract',
     'Independent Phase-0 formula diagnostics, not implemented gffx shading, a rendered scene, corpus parity, full parameter gradients or universal physical validation. The old failed gate remains saved in its historical execution.', """
white_furnace_quadrature_integrates_lambert|Check the integration method against a surface with a known energy result.|Integrate unit-albedo Lambert reflection at three resolutions; each result must equal one within 1e-12.
white_furnace_checks_single_scattering_ggx_separately|Check the microfacet contribution separately from the problematic diffuse mixture.|For a roughness-one reflector at normal view with Fresnel fixed to one, compare its integrated energy to the analytic value 1 minus log(2).
principled_candidate_satisfies_white_furnace|Historical failed check of the original material proposal.|The original local Fresnel mixture returned 1.427873492 for unit light. This historical test failed; the replacement has separate tests and this old result is not erased.
rejected_local_fresnel_mixture_still_overreflects|Keep a known bad formula as a control so we cannot hide the original problem.|Reintegrate the original local-Fresnel mixture on two grids and require its known excess of about 42.8 percent. Passing this negative control means the old formula is still rejected, not accepted.
revised_coupling_retains_energy_and_reciprocity|Check that diffuse reflection uses only the remaining light budget and does not depend on swapping viewer and light.|Use an analytic constant specular budget, three albedos including black and white, and swapped directional budgets; compare to the closed-form diffuse value.
revised_coupling_has_interior_gradients_and_safe_zero_budget|Keep the coupling usable for fitting and avoid division by zero when no energy remains.|Compare automatic and finite-difference derivatives for the coupling arguments; a fully spent budget must return zero with zero residual derivative. This does not yet test gradients through computed GGX budgets.
vndf_energy_matches_independent_hemisphere_integral|Cross-check the specular budget using two genuinely different integration methods.|Compare visible-normal quadrature at two resolutions to direct light-hemisphere integration for rough, intermediate and sharp grazing fixtures; require agreement within 0.001.
revised_principled_satisfies_white_furnace|Check that the revised material does not create light across a stated range of surface settings.|Sweep 108 combinations of roughness, specular reflectance, view angle and albedo at two precisions. Require refinement below 0.001, reflected energy at most 1.001, and white surfaces within 0.001 of one. All measurements are attached; budgets and final specular integration use different parameterizations.
revised_coupling_rejects_invalid_budgets|Reject broken inputs instead of concealing them with clamping.|Supply negative, over-one, nonfinite and inconsistent zero-residual budgets; each must raise ValueError.
gauss_quadrature_and_nonconstant_budget_have_known_integrals|Check the new integration weights and a nonconstant energy budget against independent exact answers.|Integrate eight polynomial moments, then an analytic linear directional budget and the zero-specular Lambert limit; require agreement to 1e-14.
""")


_add('python/test_light_contract',
     'Independent Phase-0 lighting recipes and documentation only. A pass is not a public gffx light API, native/CUDA parity, compilation/export or full material acceptance.', """
sh_constant_environment_returns_pi_in_every_direction|Check that a uniform environment shines equally on surfaces facing different directions.|Use the analytic constant SH coefficient and normals of different lengths; require pi irradiance in float32 and float64.
sh_directional_projection_matches_truncated_kernel_not_exact_cosine|Show exactly what the nine-coefficient approximation loses around a sharp light.|Project lights on each coordinate axis; compare the basis evaluation to an independently derived polynomial, including the 17/16 front and 1/16 back values.
sh_basis_order_and_signed_output_are_preserved|Avoid silently swapping coordinate axes or hiding negative coefficients.|Check the y,z,x first-order ordering and a negative constant environment; no output clamp is allowed.
sh_batches_do_not_mix_and_shared_inputs_accumulate_gradients|Keep different environments separate while allowing intentional shared lighting.|Use distinct RGB values per batch; then share one environment and require the derivative to accumulate both scenes.
point_inverse_square_color_direction_and_translation|Check how a point source dims with distance without losing its color or direction.|Compare one- and two-unit distances to exact RGB weights, then translate both source and receiver equally and require unchanged results.
point_coincidence_masks_only_that_pair_with_finite_zero_gradients|Handle a light exactly on a receiving point without a NaN or invented huge brightness.|Pair one coincident and one separate source; require the invalid pair's zero payload and finite zero gradients while the other pair remains valid.
point_shared_and_batched_lights_keep_explicit_axes|Keep sample points, light indices and scenes from being accidentally broadcast together.|Evaluate two scenes with four points and one light each, checking shape, masks and the exact one-to-four falloff ratio.
directional_normalizes_explicitly_without_distance_or_cosine|Make the new light's direction scale-independent and avoid applying the surface cosine twice.|Normalize nonunit directions, preserve the RGB weights, and require no ordinary solid-angle PDF for a delta light.
light_interior_gradients_match_finite_differences|Make light parameters mathematically usable for optimization away from singularities.|Compare automatic derivatives with central finite differences for SH normals/coefficients, point positions/intensities and directional vectors/weights.
light_invalid_inputs_fail_explicitly|Reject malformed or physically invalid inputs instead of silently repairing them.|Try wrong shapes/batches, mixed or integer dtypes, NaN, zero directions and negative intensity; check explicit exception categories.
light_noncontiguous_empty_and_repeat_execution_contract|Accept intentional tensor views and empty work without changing answers between repetitions.|Use strided points, empty point/light sets and repeated inputs; check output shapes and exact repeated values.
point_unrepresentable_arithmetic_is_not_silently_clamped|Avoid pretending floating-point overflow or underflow is valid physics.|Use extremely tiny and huge nonzero separations and require a representability error rather than a hidden distance floor.
light_units_and_batch_errors_cover_every_recipe|Check validation consistently across all three light recipes.|Reject non-tensors, mixed devices, zero directions, negative weights, incompatible counts/batches and overflowing point or SH outputs. A meta tensor tests device mismatch without claiming GPU execution.
lighting_demo_is_finite_and_keeps_negative_sh_values|Make the illustration finite and truthful about approximation error.|Check 16 distance samples, 181 angles, known endpoint values, negative SH ringing, invalid coincidence and the size/type of the analytic sphere image.
lighting_textbook_examples_and_evidence_links_are_present|Keep the learning explanation synchronized with the light contract.|Require point-light units, worked SH endpoints, the singularity rule and links to current tests and the finite demonstration in the generated lesson.
lighting_demo_publishes_finite_artifacts_with_honest_scope|Keep demonstrations separate from test acceptance while making them viewable.|Publish once to a temporary local test server; check demo identity, five artifact references and absence of passing-test counts.
""")

_add('python/test_learning_pages',
     'Documentation/parser checks only, not a full TeX implementation, physical iOS rendering acceptance, or material correctness.', """
mathml_subset_has_structural_fractions_scripts_and_operators|Make equations real structured mathematics rather than raw LaTeX text.|Parse fractions, roots, Greek letters, scripts and sums; inspect the MathML namespace and element hierarchy and preserved original source.
mathml_subset_supports_matrices_alignment_cases_and_text|Render small tabular mathematical expressions consistently.|Convert five environments and check row/cell structure and plain text contents.
mathml_invalid_or_excessive_input_fails_explicitly|Reject unsupported or malformed math instead of silently showing a wrong equation.|Try file/URL commands, broken groups, scripts, matrices and delimiters, plus input-size and nesting limits; require explicit ValueError results.
mathml_text_is_escaped_not_executable_markup|Keep text inside an equation from becoming HTML or a script.|Convert an image-like attack string and verify it survives only as text inside valid MathML.
mathml_styles_use_core_compatible_identifier_characters|Keep bold vectors and upright identifiers compatible with native MathML Core.|Check mathematical bold Unicode characters and the supported normal mathvariant instead of legacy inherited bold styling.
learning_article_is_generated_reproducibly_and_has_provenance|Prevent the checked-in article from drifting away from its authored source or converter.|Rebuild in memory and compare exact generated content and source hash; verify math, original results, implementation limits and evidence links without active embedded content.
learning_pages_survive_restart_without_test_execution|Keep learning material independent of experiments and pass/fail claims.|Open the article on a fresh and restarted index, verify its directory entry and date, omit its body from summaries and reject invented historical executions.
""")

_add('python/test_textbook',
     'Authored curriculum and small arithmetic checks, not exhaustive graphics correctness, physical browser acceptance or newly implemented operations.', """
textbook_has_ordered_complete_chapters_and_acyclic_prerequisites|Keep the mini-textbook readable as a connected learning path.|Require fourteen unique chapter identities and prerequisites that already appear earlier in the reading order.
textbook_chapter_is_substantial_reproducible_and_mathematically_structured|Prevent missing or stale chapters from passing as a finished textbook.|For each new chapter compare generated JSON to authored content, require explanations, exercises, evidence boundaries and parseable MathML, and enforce the article size bound.
textbook_all_links_resolve_and_assets_are_fixed_not_user_paths|Keep all textbook pages navigable and safe independently of test runs.|Read all fifteen pages from a fresh and restarted index, resolve internal links against the catalog, reject historical runs and unknown/path-traversal-like asset names.
textbook_worked_examples_match_independent_arithmetic|Catch numerical mistakes in the printed examples without borrowing graphics-kernel answers.|Compute dot product, triangle area, interpolation, camera projection, segment distance, sampling weights, a loss derivative and free fall using small standard-library arithmetic references.
textbook_index_lists_reading_order_and_implementation_boundaries|Keep the entry page honest about scope and future work.|Require every chapter title, an explicit distinction from implemented features and the future curriculum section.
textbook_catalog_refresh_preserves_dates_but_updates_titles_and_parents|Keep renamed chapters and newly classified tests in the right navigation locations after a server restart.|Seed old titles and an unassigned test in a temporary index, reload the current catalog, then require current labels and membership while preserving the original creation date.
""")

_add('python/test_shading_fit_closure', 'Finite CPU float64 fitting controls on tiny fixtures; not full-image, CUDA, optimizer convergence or real-consumer acceptance.', '''
fit_closure_has_a_finite_complete_parameter_schedule|Which learnable inputs are checked before declaring shading usable for fitting?|Require both materials and all eight continuous input groups, plus a separate spherical-harmonic coefficient objective.
fit_closure_rejects_invalid_work_before_evaluation|Can an invalid or expired fitting request accidentally start expensive shader work?|Reject unknown models, non-continuous parameter names and expired deadlines before evaluating the public chain.
fit_closure_rejects_wrong_derivative_and_non_descent|Would the qualification notice a wrong gradient or an update that fails to improve the image?|Inject inconsistent finite differences and unchanged loss, and require explicit rejection.
saved_fit_closure_requires_exact_evidence|Do all saved fitting measurements still match the tested implementation and resource limits?|Verify the exact report hash, wheel and runtime identities, seventeen-case schedule, monitored worker, fixed call count, two-scale derivatives and fresh descent.
each_saved_shading_parameter_has_checked_fresh_descent|Does this particular parameter have an accurate gradient that improves the image?|Show its saved two-scale finite differences and before/after objective in separate named checks, after strict worker and installed-wheel replay.
''')

_add('python/test_shading_execution_boundary', 'Negative capability tests only. No accepted compiled graph, serialized model, default graph-break mode or real-time inference claim.', '''
shading_fullgraph_rejection_never_calls_a_fallback_backend|Does a fullgraph request accidentally pretend eager shading is compiled?|Require data-dependent-check rejection before invoking any backend, then verify unchanged eager output and negative-albedo rejection.
shading_export_rejection_preserves_eager_domain_checks|Can export silently remove the shader input checks?|Try both strict and non-strict export, require explicit rejection, then verify finite eager output and nonfinite-input rejection.
''')

_add('python/test_shading_cuda_closeout', 'Coordinator controls do not themselves run the CUDA physical cohort or grant catalog admission.', '''
cuda_closeout_stops_after_first_failed_stage|Will one failed CUDA gate stop the remaining work?|Inject a failed first stage and require one invocation only, with the stopped coordinator preserved.
cuda_closeout_rejects_changed_wheel_before_device_execution|Are results bound to the actual artifact rather than just a package version?|Supply a different wheel digest and require identity rejection before CUDA work.
cuda_closeout_restores_cpu_memory_policy_after_failure|Can the special CUDA allowance accidentally raise CPU limits?|Interrupt the CUDA worker launch and require the original CPU profile object to be restored, while observing only the approved 2 GiB CUDA-worker override.
''')

_add('python/test_environment_image_adjoint', 'CPU float64 method controls only; two tiny material comparisons are not whole-image qualification.', '''
angular_jacobian_keeps_cross_channel_derivatives|Can we keep a small lighting graph without losing how one colour channel influences another?|Use two deliberately colour-mixing lighting chunks; compare the complete 3x3 Jacobian and the final loss derivative to a direct automatic derivative.
angular_jacobian_rejects_invalid_work_before_callback|Will invalid input start lighting work anyway?|Try an empty schedule, expired deadline, wrong precision and wrong shape; require rejection before any callback executes.
environment_image_adjoint_matches_two_pass_texture_gradient|Does the memory-bounded method give the same source-texture derivative as the earlier implementation?|For each material, compare complete RGB, every source derivative and exact forward/reverse call counts on one small environment-lighting fixture.
environment_image_adjoint_rejects_work_before_material|Can a requested sampling order silently exceed the work budget?|Supply an unsupported angular order and require rejection before the public material is called.
''')

_add('python/test_environment_image_schedule', 'Scheduling and independent affine-reference controls; no new public whole-image render or gradient measurement.', '''
environment_schedule_partitions_every_pixel_under_both_limits|How do we divide an image without losing pixels or overloading one worker?|Partition mixed sampling orders; require every pixel exactly once and both the material-call and point-count limits, including both sampling offsets.
environment_schedule_rejects_unbounded_orders|Can malformed or oversized sampling instructions reach the worker schedule?|Reject unsupported orders, booleans, floats, empty work and more pixels than the fixed image contains.
environment_schedule_checks_progress_hash_before_json|Are we actually using the previously validated forward evidence?|Require a matching immutable byte hash before attempting to parse the progress record.
environment_schedule_rebases_independent_colour_affine_reference|Can the independent lighting reference be reused for a different colour texture without borrowing public outputs?|Check by hand that subtracting the original diffuse colour response yields the correct intercept and independently predicted target colour.
''')

_add('python/test_environment_image_runner', 'Assembly and launch-safeguard controls, not evidence that the complete image workers have passed.', '''
image_adjoint_assembly_weights_unequal_tiles_by_pixel_count|Does a small final tile accidentally count as much as a large tile in the image loss?|Combine a two-pixel tile and a one-pixel tile; require the loss and gradient weighted by pixel count, with the original pixel order preserved.
image_adjoint_assembly_rejects_partial_or_invalid_evidence|Could incomplete, duplicated or invalid pixels be reported as a complete image?|Inject a gap, overlapping range, missing work and nonfinite loss; require explicit rejection.
image_adjoint_worker_rejects_wrong_schedule_hash_before_loading|Can a worker silently switch to another sampling schedule?|Give corrupted bytes with a different expected hash and require rejection before loading the record.
image_adjoint_next_worker_reserves_watchdog_time|Will the coordinator start another worker when too little aggregate time remains?|Check that a fresh profile can launch, but the last 180 seconds are reserved for the watchdog rather than starting unfinishable work.
''')

_add('python/test_environment_image_evidence', 'Replay of saved finite CPU colour-fitting evidence only. Partial checkpoints remain partial; no public shading is rerun by these checks.', '''
environment_image_evidence_rejects_unknown_profile|Can an unknown material or environment acquire a result page by changing its name?|Reject profiles outside the four predeclared image schedules before reading evidence.
environment_image_evidence_rejects_hash_before_parsing|Do we verify the actual saved bytes before interpreting their claims?|Reject a different digest before parsing the cohort record.
environment_image_saved_prefix_is_truthfully_partial_or_complete|How much of this image has really been checked so far?|Replay every listed immutable worker, its numerical reference comparisons and resource limits; distinguish a valid prefix from a complete two-measurement image.
environment_image_evidence_rejects_rehashed_false_claims|Would changing a report and calculating a new hash hide missing work?|Forge source, schedule, call-count, time allowance, ownership and completion fields; all must still be rejected by semantic checks.
environment_image_complete_request_rejects_a_valid_prefix|Can valid partial progress satisfy a request for a finished image?|Keep a genuine first worker but mark the record partial; explicit complete-image replay must reject before numerical assembly.
environment_image_complete_profile_replays_all_pixels_and_fresh_descent|Does this material and environment really have complete image-gradient evidence?|Replay all104 workers and both6536-pixel images, their independent comparisons and finite differences, then require both offset losses to decrease after the shared texture update under the original memory limit.
''')

_add('python/test_environment_image_demo', 'Finite display tests, not new lighting accuracy measurements. Images are enlarged for readability without adding measured pixels.', '''
environment_image_preview_marks_uncomputed_pixels_explicitly|Can you tell which parts of the displayed image have actually been evaluated?|Check a tiny ownership fixture: verified pixels match the displayed target, unknown surface pixels are grey, and background is dark.
environment_image_preview_rejects_invalid_ownership_or_values|Could the preview conceal duplicated pixels, NaNs or an inflated completion count?|Inject each defect and require rejection before creating display images.
environment_image_publisher_uses_supported_provenance_and_finite_demo|Will the actual dashboard accept the preview without mistaking it for passing tests?|Publish to a real temporary server; require supported producer metadata, five finite artifacts, demo status and no passing-test counts.
environment_image_gallery_requires_all_four_complete_profiles|Can the final gallery quietly omit a material, duplicate a result or call partial work complete?|Reject missing, repeated or incomplete profiles before connecting to the server.
environment_image_gallery_publishes_every_comparison_without_test_claims|Will all four before/after comparisons stay visible together?|Use a temporary server to verify twelve uniquely named images, four evidence records, four loss plots and scope text, with no invented test passes.
''')

_add('python/test_principled_cuda_catalog', 'Pure source catalog and receipt controls; these tests alone do not execute the installed GPU implementation.', '''
cuda_catalog_uses_a_distinct_named_profile|Can the CUDA table have its own acceptance without changing what the CPU name means?|Require different resource/profile IDs and reject the wrong precision or device; generic acceptance remains CPU-only.
cuda_catalog_rejects_changed_physical_identity|Could another table borrow the accepted CUDA label?|Change the payload hash, model or generator and require rejection despite otherwise matching metadata.
cuda_catalog_reuses_exact_payload_without_duplicating_data|Does supporting CUDA add another large table or runtime download?|Read both named routes and require identical bytes, with no duplicate CUDA payload file in the source tree.
cuda_catalog_ships_an_exact_finite_acceptance_record|What evidence actually supports the named CUDA resource?|Verify the exact receipt hash, parent payload, five qualification-stage hashes, counts and explicit unqualified modes.
''')

_add('python/test_principled_cuda_bundle', 'Installed CUDA float64 checks on real hardware. Finite acceptance only, not all-GPU or continuous-domain proof, whole-image CUDA gradients or sustained real-time performance.', '''
named_cuda_bundle_has_exact_installed_receipt_and_local_tensors|Does the installed named bundle really contain the qualified record and CUDA constants?|Verify the packaged receipt bytes and accepted metadata, identical numerical envelope and every internal constant on the requested CUDA device.
named_cuda_bundle_does_not_promote_generic_transfer_or_loading|Does the new bundle silently turn old research transfers into accepted resources?|Require default generic CUDA loading/transfer to reject and explicit research transfers to remain experimental, even when starting from an accepted object.
named_cuda_bundle_rejects_a_different_lane|Can the CUDA-specific name accidentally select CPU or unsupported float32 data?|Request each wrong lane and require an explicit error.
named_cuda_bundle_matches_experimental_and_cpu_query_partials|Does assigning the new acceptance identity change any query mathematics?|Compare directional-plus-mean values and all three partials: exact equality to the experimental CUDA path and tight parity with CPU.
named_cuda_bundle_material_needs_no_experimental_flag|Can a caller use the named qualified resource normally without changing the shader output or its derivatives?|Evaluate a complete material without the research flag and require exact CUDA output/all-input derivative equality with the explicit experimental route.
''')

_add('python/test_cuda_bundle_probe', 'Coordinator controls only. Separate installed worker reports own the numerical CUDA qualification.', '''
cuda_bundle_probe_restores_cpu_limits_after_interruption|Can CUDA's larger host allowance leak into later CPU work?|Interrupt a simulated worker launch, verify only the approved host limit changes, then require restoration of the original CPU policy object.
cuda_bundle_probe_does_not_admit_after_parent_failure|Can a failed physical qualification still produce named admission?|Fail the parent cohort and require a stopped record without starting the named-bundle stage.
''')

_add('python/test_shading_evidence_bytes', 'Source publication safety only; this is not a numerical or platform qualification.', '''
shading_hash_bound_files_preserve_exact_checkout_bytes|Could saving the source in Git invalidate otherwise unchanged test evidence?|Require explicit byte-preserving rules for the exact source files bound by current image/CUDA reports and lesson fingerprints; no repository-wide normalization or Git publication occurs.
''')


_add('python/test_budget_default_device', 'Explicit device ownership checks; a meta default is a hardware-free way to expose accidental ambient allocation.', '''
budget_explicit_cpu_overrides_ambient_meta_device|Can another library's default tensor device override the CPU device you explicitly requested?|Construct the bundle, an envelope and a transfer inside a meta-device context; require real CPU constants, unchanged metadata and bit-identical coefficients against ordinary CPU construction.
''')


_add('python/test_scene','Minimal eager scene profile, not release or whole-gffit acceptance.', '''
scene_facade_and_empty|Can applications start with import gffx and an empty scene?|Check the facade, empty packed tensors and two empty frame boundaries.
scene_hierarchy_packing_and_existing_validation|Does moving a parent position its children correctly?|Compare three-level transforms with hand-computed coordinates in both precisions, then verify indices and existing CPU mesh validation.
scene_attribute_domains_and_missing_mask|Do UVs and other values stay attached to the correct corners?|Independently expand four attribute domains, check missing masks and preserve source gradients.
scene_frames_and_batched_attributes|Can frames share topology without mixing colours or positions?|Check per-frame transforms and colours, globally rebased faces and exact boundaries.
scene_bindings_snapshot_and_tensor_borrowing|Are bindings retained without surprising ownership changes?|Check typed material/light/camera payloads and copied container membership, borrowed tensors and nonaliasing outputs.
scene_normals_nonuniform_scale_and_reflection|Do normals stay consistent under scale and mirroring?|Compare signed inverse-transpose transport with a hand-calculated normal.
scene_strides_defaults_repeat_and_no_alias|Are noncontiguous tensors and ambient defaults safe?|Flatten strided CPU data inside a meta-device context and verify repeatability and output ownership.
scene_rejects_invalid_input|Will invalid scene data fail clearly before rendering?|Reject duplicate/cyclic names, missing references, invalid devices, precision, transforms and attribute shapes.
scene_mutation_revalidated_and_unknown_options_rejected|Can later edits bypass validation?|Reject incompatible options and a borrowed face tensor changed to a negative index.
scene_empty_mesh_and_shared_gradients|Do empty objects disturb packing or shared gradients?|Include an empty mesh, check offsets and require two references to accumulate the expected vertex derivative.
scene_composed_singular_transform_rejected|Can individually valid transforms become numerically invalid when combined?|Compose tiny float32 scales and require explicit rejection of the underflowed world determinant.
scene_vertex_sequence_seams_and_deep_hierarchy|Can a deep scene preserve animated vertices and texture seams?|Traverse1100 parents without recursion, pack two geometry frames and compare per-corner UVs exactly.
scene_cross_mesh_channel_mismatch_and_constructor_types|Can incompatible objects or attributes sneak into the scene?|Reject inconsistent channel counts and invalid evaluator, camera and prim types.
''')
_add('python/test_scene_integration','Finite local profiles; CPU mesh.validate is not available on CUDA. No export or sustained throughput claim.', '''
scene_integration_raster_matches_manual_and_frames|Does flatten produce what the existing renderer expects?|Compare three frames with manual rasterization, including face IDs and all floating fragment outputs.
scene_integration_image_gradient_and_fresh_descent|Can an image error adjust an object position through the scene?|Finite-difference check smooth translation and require a fresh gradient step to lower rendered colour error.
scene_integration_parent_and_vertex_gradcheck|Do derivatives reach geometry and parent transforms?|Compare packed vertex derivatives with finite differences for parent translation and original vertices.
scene_cuda_inherited_packing_and_gradients|Does the scene stay on GPU and retain gradients?|Compare CPU/CUDA packing, rasterization and derivatives without an implicit CPU-validator copy.
scene_budget_synthetic_flame_scale|What does preparing one or sixty frames cost?|Measure five samples after warmup at synthetic 5023-vertex/9976-face cardinalities and count output bytes; no licensed data.
''')
_add('python/test_scene_presentation','Presentation controls, not numerical acceptance by appearance.', '''
scene_demo_calls_public_flatten_and_is_bounded|Are the demos using the real scene implementation?|Observe public flatten calls, bound the clip to twelve frames and require the small fitting example to reduce loss.
scene_every_test_has_authored_explanation|Will every scene test explain its purpose?|Enumerate scene test functions and require authored descriptions rather than guessed explanations.
scene_textbook_and_catalog_agree|Can readers find the scene mathematics?|Require stable chapter, test-page and demonstration identifiers together.
''')

_add('python/test_scene_probe','Coordinator regression, not numerical device evidence.', '''
scene_worker_selects_exact_nodes_not_filename_substrings|Can a CPU worker accidentally select GPU tests again?|Require exact disjoint node IDs for integration, GPU and budget modes rather than matching the filename.
''')

_add('python/test_scene_showcase','Finite presentation checks, not physics, Cycles, performance or numerical shading acceptance.', '''
bevels_have_outward_flat_normals_and_no_corner_averaging|Why do these cubes have clean edges?|Check every bevel triangle has outward winding and unit face normals; separate corner vertices prevent accidental smoothing across creases.
public_scene_hierarchy_moves_only_the_owned_branch|Do the visible arms describe real parent relationships?|Observe public Scene.flatten and compare world transforms: branch motion leaves the turquoise anchor and stage fixed; whole-parent motion carries the anchor.
view_transports_contiguous_explicit_types|Are display buffers sent in the expected format?|Convert an explicit float64 input to the viewer float32 contract without changing its shape.
publication_has_colours_normals_and_finite_checkpoint_windows|Will the phone receive colour, crisp edges and bounded history?|Capture the publisher calls: every static and animated mesh carries colours/normals, with two24-frame checkpoint windows and no continuous producer.
all_showcase_checks_have_authored_explanations|Can you understand each new presentation check?|Require an authored purpose and method for every test in this module.
tiled_render_matches_full_native_rasterization|Does faster presentation rendering preserve native pixels?|Compare tiled and untiled native rasterization on overlapping triangles, including visibility and interpolated colour with a fixed2e-6 float32 tolerance. This does not qualify a new public rasterizer.
''')

_add('python/test_m1_lessons', 'Documentation registration and fingerprint checks, not numerical operation evidence.', '''
m1_lessons_are_registered_in_dependency_order|Make the new mathematics chapters discoverable in a sensible reading order.|Require each M1 chapter in the textbook registry and require prerequisites to precede their dependents.
m1_textbook_introduction_separates_library_progress_from_consumers|Does the textbook distinguish implemented library mathematics from demonstration scripts and other projects?|Check that the introduction names native deformation, keeps consumer implementations separate and does not call the bouncing-ball illustration a physics solver.
m1_lesson_fingerprint_covers_modular_chapters|Prevent edited chapter sources from retaining a stale generated-article identity.|Modify each modular chapter in a temporary source fixture and require the source fingerprint to change.
''')
_add('python/test_m1_capability_inventory', 'Linked primitive inventory only; not GPU qualification, performance or universal operation support.', '''
native_operation_count_matches_public_primitive_inventory|Keep capability metadata from claiming the completed CPU core still contains no operations.|Check that native transport reports the thirteen foundation plus five animation primitives rather than its old zero placeholder.
support_status_separates_m1_local_evidence_from_release_support|Does the public support page describe what we have actually tested?|Require current M1 operation and artifact context while keeping consumer, device, release and inherited-evidence limits explicit.
''')
_add('python/test_m1_reporting', 'Dashboard navigation only; no numerical acceptance is inferred.', '''
m1_routing_preserves_specific_older_pages|Keep earlier shading results on their established pages while adding M1 features.|Resolve old image-loss fixtures and new animation storage/gradient fixtures to their exact feature pages.
''')

_add('python/test_testing_artifact_identity', 'Caller-supplied identity labels require a separately verified installed artifact; no automatic certification.', '''
test_session_reports_only_valid_artifact_identity|Which exact installed build produced these test results?|Feed valid and malformed artifact labels to the reporting session and require only the valid SHA256 label in provenance.
''')
_add('python/test_native_dependency_scan', 'Static dependency guard, not dynamic memory or CPU/CUDA numerical qualification.', '''
native_fragment_dependency_scan|Can internal C fragments hide an unwanted allocation or dependency?|Scan a harmless owned fragment and a synthetic fragment containing malloc; admit the first and explicitly reject the second.
''')

_add('packaging/test_image_build_contract', 'Build configuration only, not numerical or cross-platform acceptance.', '''
image_strict_arithmetic_reaches_every_core_directory|Do all three core builds preserve the same colour-math rounding rules?|Configure compiler-free targets in parent and child CMake directories, then inspect each source option list for MSVC and non-MSVC settings.
image_ptx_tracks_transitive_transfer_headers|Will changing shared colour arithmetic rebuild the GPU kernels?|Follow the scalar header's local includes and require every dependency on the actual PTX compilation rule; this checks rebuild wiring, not numerical accuracy.
image_ptx_rejects_missing_boundary_dependency|Would the build check catch an outdated GPU kernel?|Remove the boundary header from a copy of the build rule and verify the dependency checker rejects that deliberate defect.
''')

from .image_guides import GROUPS as _IMAGE_GROUPS
from .deferred_guides import GROUPS as _DEFERRED_GROUPS
from .animation_guides import GROUPS as _ANIMATION_GROUPS

for _guide_path, _guide_limits, _guide_rows in (*_IMAGE_GROUPS, *_DEFERRED_GROUPS, *_ANIMATION_GROUPS):
    _add(_guide_path, _guide_limits, _guide_rows)


def explain(nodeid, check=None):
    """Unknown identifiers are data, never filesystem paths or inferred evidence."""
    key=nodeid.split('[',1)[0]
    value=GUIDES.get(key) if check is None else CHECK_GUIDES.get((key,check))
    return dict(authored=value is not None, basis='current author guide', **(value or dict(
        purpose='No authored explanation is available for this test or named check yet.',
        method='Use the source identifier and reported diagnostics to inspect it; the dashboard does not infer intent from a hash or assertion result.',
        limits='An unexplained test result is not a claim of feature acceptance. Add a reviewed guide before treating this as a maintained inspection page.')))

_add('packaging/test_opaque_shading_intervals','Independent finite quadrature controls. Bounds are conditional on the declared smooth field and outward arithmetic; no full renderer admission.', '''
validated_triangle_polynomials_enclose_exact_integrals|Does the reference contain known integrals rather than just agree with another grid?|Compare triangle polynomial moments of degrees zero through six with exact rational values.
validated_triangle_parameter_derivative_is_integrated_separately|Can a correct image hide an incorrect gradient?|Integrate the analytic parameter tangent separately and compare its exact rational answer.
validated_interval_arithmetic_and_jet_transcendentals|Do rounded operations preserve enclosures?|Check division, roots, logs and derivative coefficients against known constants and reject a zero-containing denominator.
validated_quadrature_detects_sample_missed_polynomial|Could all samples miss a nonzero function?|Use a polynomial zero at all coarse Gauss nodes; require budget refusal first, then enclose its strictly positive exact integral.
validated_quadrature_refuses_invalid_domains_limits_and_cancellation|Can unfinished work become a plausible reference?|Reject bad tolerance, exhausted work, invalid log domains and cancellation without returning a last estimate.
validated_triangle_orientation_and_tiny_positive_area|Are tiny surfaces lost or winding-dependent?|Enclose the exact small positive triangle area under both orientations.
validated_jet_keeps_interval_constant_separate_from_formal_increment|Can interval uncertainty contaminate formal differentiation?|Retain the original log value interval and independently enclose derivative endpoints.
validated_triangle_mixed_coordinates_and_moving_geometry|Does changing geometry differentiate the integration measure?|Check an exact mixed polynomial and the analytic moving-triangle area derivative; this alone is not visibility-gradient acceptance.
validated_gauss5_rule_encloses_exact_high_degree_controls|Is the higher-order rule independently checked?|Compare six exact rational polynomial moments through degree ten.
validated_gauss5_refuses_sample_missed_degree_ten|Can a higher-order grid falsely declare zero?|Use the squared Gauss-node polynomial; refuse a single-cell shortcut and recover its exact positive integral with subdivision.
order_ten_mixed_jets_enclose_independent_series|Are the derivatives used to bound gradient error themselves checked?|Compare every spatial order through ten and its first parameter tangent with independent reciprocal, logarithm and square-root series.
''')
_add('python/test_opaque_principled_integral','One smooth fixed-roughness perspective pixel. Separate image and first-tangent bounds; no full scene, CUDA or general texture/visibility-gradient qualification.', '''
principled_interval_point_contains_independent_decimal_formula|Does the interval shader evaluate the intended complete material?|Contain a separate 90-digit GGX/Smith/Schlick/reciprocal-matte formula multiplied by one illumination cosine.
principled_pixel_bound_detects_vertex_colour_surrogate|Can shading only at vertices stand in for nonlinear pixel integration?|Certify the actual perspective-varying material integral and one normal/albedo tangent at unchanged error limits; reject the vertex-shaded affine surrogate.
principled_reference_refuses_knot_and_singular_profile|Does the reference admit nonsmooth or undefined cases without a bound?|Refuse an unpartitioned interpolation-knot crossing and zero normal vector.
''')
_add('python/test_opaque_principled_comparison','Installed-wheel point shader integrated over the saved reference partition. Numerical comparison is separate from the independent error enclosure and does not install a nonlinear scene path.', '''
installed_principled_weighted_integral_matches_independent_enclosures|Does the installed shader agree with the independently bounded pixel and tangent?|Verify saved evidence identity, use 1100 weighted points in serial groups of four, differentiate actual installed calls and compare RGB and gradient separately with the reference intervals.
principled_interval_tangent_agrees_with_independent_decimal_differences|Is the interval shader tangent corroborated independently of its own algebra?|Compare two tiny-step 90-digit central differences of the separate material formula with the analytic interval tangent; this is corroboration, not the integral error proof.
exact_visibility_partition_keeps_hidden_and_subpixel_mass|Does the nonlinear integration wrapper preserve exact visibility ownership?|Use constant fields to compare visible colours with the independent rational polygon oracle, requiring zero hidden contribution and positive subpixel mass.
''')

_add('packaging/test_opaque_moving_reference','Local moving visibility with exact polynomial branch certificates. No general topology-transition, production scene or CUDA admission.', '''
moving_silhouette_integrates_area_field_and_measure_derivatives|Does a moving edge change both visible area and sampled shading?|Compare exact polynomial pixel integrals and analytic field/Jacobian tangents; require the nonzero boundary contribution that frozen visibility loses.
moving_depth_intersection_matches_closed_form|Does exchanging the frontmost surface produce the correct colour gradient?|Move a depth boundary at two positions and compare exact piecewise polynomial integrals and derivatives.
moving_hidden_surface_and_background_have_exact_derivatives|Can an invisible surface contribute a false gradient?|Move overlapping silhouettes, require zero hidden contribution and include the exposed background derivative.
moving_subpixel_triangle_retains_positive_area_and_gradient|Does tiny geometry survive differentiation?|Require exact positive area and expansion derivative for a triangle smaller than a pixel.
moving_geometry_agrees_with_independent_rational_differences|Does the new moving geometry agree with the earlier independent oracle?|Check both windings against exact nominal colours and two rational finite differences in a closed-form linear fixture.
moving_branch_certificate_rejects_first_order_invisible_transition|Can a zero first derivative hide a visibility change?|Reject a nonzero quadratic predicate vanishing at the expansion point; independently check the certificate's rational sign-preservation inequality.
moving_visibility_refuses_tie_changes_and_collapsing_geometry|Will a nondifferentiable visibility configuration be silently accepted?|Refuse changing depth ties and a triangle that opens from zero area.
moving_reference_preserves_work_and_cancellation_guards|Does moving-region work retain its limits?|Reject zero work allocation and propagate cancellation before integration.
moving_shared_edges_do_not_invent_internal_coverage_gradients|Will translating a triangulated surface create internal seam gradients?|Check both square diagonals and a subdivision using the same screen field, with exact full coverage and zero net geometry derivative.
general_moving_geometry_matches_independent_affine_oracle|Do simultaneous x,y and depth changes retain the full derivative?|Compare two small exact-rational finite differences with the analytic tangent and independently replay every topology sign bound.
empty_moving_reference_validates_tolerance_and_returns_exact_zero|Can empty input bypass validation?|Reject zero tolerance, then require exact zero image,coverage and derivatives for a valid empty scene.
integrated_attached_field_translation_cancels_false_shading_gradient|Does a material move with its surface?|Translate an entirely visible triangle and require zero integrated derivative for its attached colour field; keeping that field fixed in screen coordinates must produce a detectable wrong gradient.
''')
_add('python/test_opaque_moving_principled','Two smooth,y-independent nonlinear moving-edge fixtures and local material transport. Reference integration bounds exclude native binary64 rounding; no full scene,Cuda or arbitrary material admission.', '''
nonlinear_moving_pixel_matches_boundary_formula_and_installed_gradient|Does full Principled shading retain moving visibility gradients?|Compare bounded triangle integration with independent 90-digit boundary formulas and a different Gauss16 installed-wheel integral/autodiff, checking RGB and gradients separately.
installed_affine_visibility_gradient_matches_new_exact_geometry|Does the installed opaque primitive agree with the independent moving-region geometry?|Compare actual installed affine-colour output and autograd geometry derivatives to the new exact reference at silhouette and depth boundaries.
moving_perspective_field_transports_attributes_with_the_triangle|Do perspective-interpolated attributes stay attached to moving geometry?|Translate both query and triangle; require cancellation of the attached field's tangent and reject the false tangent of a screen-fixed field.
''')

_add('packaging/test_opaque_material_reference','Finite independent material partition and derivative controls; no general renderer admission.','\nexponential_mixed_jet_includes_order_ten_and_tangent|Are exponential derivative bounds correct?|Compare all ten spatial orders and the first parameter tangent with independent analytic coefficients.\nbranch_selection_refuses_uncertainty_instead_of_picking_midpoint|Can an uncertain limiter branch silently become a result?|Reject overlapping branch intervals and an absolute-value zero crossing.\nmaterial_boundary_pixel_includes_moving_interface_flux|Does a moving material interface contribute to the pixel gradient?|Integrate both regions and compare exact continuous-field and discontinuous-jump boundary formulas.\nbilinear_texture_integral_and_texel_uv_derivatives|Are texture values and parameter derivatives integrated separately?|Compare exact bilinear moments and analytic texel/UV tangents on a unit pixel.\ntexture_kink_is_partitioned_even_when_point_samples_miss_it|Can a narrow texture feature vanish between samples?|Partition a feature only 1/65536 pixel wide and require its exact nonzero contribution.\npartition_refuses_topology_events_and_work_exhaustion|Will a partial partition masquerade as a reference?|Refuse changing topology and capacity exhaustion, and propagate cancellation.\ncentred_jets_enclose_every_derivative_and_cancel_dependency|Can centred derivative bounds safely reduce interval dependency?|Contain analytic derivative values with natural and mean-value enclosures while retaining exact cancellation controls.\ntaylor_envelope_remainder_cannot_drop_unseen_seventh_order_term|Could the Taylor approximation silently discard a real contribution?|Use a seventh-order negative control that vanishes in the lower polynomial and require containment by the remainder.\nnormalized_texture_cells_integrate_wrap_and_uv_motion|Are clamp and repeat cells defined in normalized UV coordinates?|Integrate explicitly selected bilinear cells and compare exact values and UV motion across texel-centre boundaries.\nperspective_texture_boundary_motion_has_exact_quotient_gradient|Does a perspective texture boundary move correctly?|Compare the homogeneous isoline location and derivative with exact rational quotient formulas.\n')

_add('python/test_opaque_material_principled','Independent bounds concern integration of the stored loss resource; native rounding and physical-table approximation remain separate. Smooth certified branches only.','\nroughness_point_and_first_derivative_match_independent_formula|Does full Principled roughness differentiation match separate mathematics?|Compare point radiance and tangents against a separately evaluated 90-digit shader and two tiny finite differences.\nspatial_roughness_has_separate_value_and_gradient_error_bounds|Can spatially varying roughness be integrated with an explicit error bound?|Use centred seventh-derivative enclosures and validated Gauss3 integration; check the gradient with the fundamental theorem and installed Gauss16 independently.\nroughness_branch_diagnostic|Which interpolation limiter branches remain certified over the roughness interval?|Construct all 512 angular controls with strict interval branch checks and preserve failed earlier trials.\ntextured_material_transition_matches_exact_formula_and_installed_calls|Do discontinuous materials, UV motion and texel changes have correct pixel derivatives?|Combine exact bilinear moments with a separate 90-digit Principled formula on both sides, then compare installed shader/texture calls.\n')

_add('python/test_native_opaque_quadrature','CPU geometry quadrature and selected-topology first VJP. This primitive alone does not certify an arbitrary nonlinear shader.','\nnative_quadrature_matches_exact_affine_visibility_and_geometry_gradient|Does native visible-region integration agree with independent exact geometry?|Contract emitted barycentrics and area weights with affine colour; compare exact pixel values and geometry differences including a depth intersection.\nnative_quadrature_moving_material_cut_preserves_boundary_flux|Are both sides of a moving material cut retained?|Assign distinct radiance by side and require exact image 2.5 and derivative -2.\nnative_quadrature_all_geometry_and_cut_coordinates_gradcheck|Are all quadrature coordinate derivatives connected?|Check the full selected-topology Jacobian for geometry and material-cut numerators against finite differences.\nnative_quadrature_empty_padding_capacity_and_budget_refusals|Does fixed-capacity output remain unambiguous and bounded?|Check empty and unused slots, excessive candidates and insufficient allocation budgets; refuse instead of truncating.\nnative_quadrature_preserves_subpixel_area_at_large_origins|Does subtracting large pixel origins erase small triangles?|Check positive tiny area in float32/float64 at origins 0,1024 and65536.\nnative_quadrature_refuses_nonfinite_second_derivatives_and_cuda|Are unsupported paths explicit?|Reject nonfinite inputs, unsupported second derivatives and CUDA without silently copying to CPU.\n')

_add('python/test_opaque_nonlinear_scene','Small CPU float64 Principled/directional scene composition. Dense/CUDA, arbitrary nonsmooth materials and final showcase qualification remain open.','\nopaque_scene_integrates_nonlinear_principled_against_independent_pixel|Does the complete scene render the nonlinear pixel rather than vertex-shaded interpolation?|Compare the installed camera/material/light composition with an independent outward-rounded pixel integral and require full opaque coverage.\nopaque_scene_no_interior_transparency_across_diagonals|Do texture and roughness fields reintroduce internal mesh seams?|Render either square diagonal with texture, roughness or both; compare images under unchanged tolerances and require alpha one.\nopaque_scene_camera_material_and_texture_gradients_are_not_detached|Can scene composition sever a parameter derivative?|Move geometry,camera,UV,texels,roughness,normal,F0 and light together; require a nonzero first derivative and two finite-difference confirmations. This combined direction is not every independent partial.\n')

_add('python/test_opaque_nonlinear_scene','Small CPU float64 profile; individual directional checks are corroboration and do not certify arbitrary scene integration.', '''
opaque_scene_independent_parameter_directions|Can individual parameter paths be disconnected despite a combined-direction pass?|Check eight independent geometry,UV,texel,roughness,F0,light,normal and camera directions against two finite differences each.
opaque_scene_saves_texture_roughness_demonstration|What does the actual connected scene renderer produce?|Save all16 RGB pixels and coverage of a4x4 textured/varying-roughness Principled surface; require full coverage and finite values. The magnified image is an engineering demonstration,not final showcase acceptance.
''')

_add('python/test_opaque_streaming','Finite CPU float64 scenes; exact partition controls do not establish universal quadrature accuracy or dense-scene speed.', '\nopaque_stream_admits_formerly_refused_eight_square_and_matches_windows|Can bounded windows render the formerly refused scene?|Compare every RGB and mask value for automatic,one-pixel and three-pixel windows on the8x8 texture/roughness fixture.\nopaque_stream_replayed_gradient_is_tile_partition_independent|Does changing the forward window alter backward accumulation?|Replay pixels in one fixed order and require exactly equal vertex gradients across three window caps.\nopaque_stream_retains_inputs_not_per_window_graphs_and_checks_mutation|Does backward retain an entire frame of intermediate graphs?|Inspect saved tensors and require input-mutation detection before replay.\nopaque_stream_validates_tile_and_total_output_budget|Can windowing evade the total memory allowance?|Reject invalid caps,insufficient workspace and a frame whose output reserve alone is too large.\nopaque_stream_batched_depth_material_and_subpixel_windows_match_resident|Do batches and overlapping materials change at window borders?|Compare two batched scenes with a small foreground surface against the resident installed path for both window choices.\nopaque_stream_replayed_geometry_gradient_agrees_with_two_differences|Does replay reconnect moving geometry to the image?|Compare the streamed full objective derivative with two independent finite-difference spacings.\nopaque_stream_refuses_mask_only_tile_argument|Can a tile option be silently ignored?|Require a clear rejection when a colour-only option is supplied to mask-only rendering.\n')
_add('python/test_opaque_cuda','Finite device-owned quadrature and first derivatives. CPU/CUDA equality is conformance,not an independent image-integration proof.', '\nopaque_cuda_quadrature_matches_cpu_with_material_and_depth_boundaries|Does device quadrature preserve CPU geometry and derivatives?|Use depth overlap and a material cut in float32/float64 on a nondefault CUDA stream; require identical owners,samples and first VJPs.\nopaque_cuda_rejects_invalid_device_data|Are bad device inputs validated before use?|Reject out-of-range indices and nonfinite geometry or cut tensors without a host geometry fallback.\nopaque_cuda_scene_keeps_device_ownership_and_matches_independent_cpu_anchor|Does the connected textured scene keep its computation on the chosen device?|Compare CPU/CUDA RGB,mask and vertex VJP on the existing texture/roughness fixture. Its conformance is separate from the earlier independent pixel reference.\n')
_add('python/test_opaque_material_branches','Piecewise-constant branch fixtures only. General varying-roughness,angular-knot and horizon crossings remain unresolved; physical table approximation is separate.', '\nopaque_material_branch_pixel_and_boundary_flux|Do wider material branches preserve both colour and moving-interface gradients?|Use roughness0.15..0.95,F0 endpoints,two-sided and unlit cases; compare exact partition integrals and90-digit formulas with installed quadrature/shading.\nopaque_material_unresolved_horizon_and_angular_crossings_refuse|Can an uncertain material branch be certified from samples?|Require explicit refusal of an interval spanning a horizon or angular interpolation knot.\n')
_add('packaging/test_ptx_embedding','Build representation check only; actual compiler memory and driver execution require their separate supervised gates.', '\nptx_chunked_literals_preserve_every_byte_and_nul|Can smaller compiler initializers alter the GPU program?|Exercise partial,full and multiple2048-byte rows with every byte value; require exact payload plus one explicit NUL and bounded literal lengths.\n')
_add('python/test_native_opaque_quadrature','First-derivative primitive; CUDA qualification is recorded separately.', '\nnative_quadrature_refuses_nonfinite_second_derivatives|Are unsupported derivatives and invalid inputs explicit?|Reject nonfinite CPU inputs and higher derivatives; device support is tested in the CUDA family.\n')

_add('python/test_opaque_cuda_diagnostic','Diagnostic capture only. A passing capture does not mean the measured device gradients agree.', '\ncapture_opaque_cuda_scene_adjoint_stages|Where do CPU/CUDA scene values or gradients first differ?|Capture installed intermediate values and adjoints on the executing autograd thread; report exact comparison,maximum difference and repeat-device stability without changing the computation.\n')
_add('python/test_opaque_streaming','Finite smooth normalization derivative control,not a full-renderer proof.', '\nopaque_perspective_normalization_explicit_vjp_gradcheck|Does explicit reduction order preserve the mathematical derivative?|Check all raw-weight and reciprocal-depth partials at two smooth points against finite differences and require the defined three-term forward sum.\n')

_add('python/test_opaque_cuda','Direct installed C ABI with caller-owned CUDA buffers; finite boundary controls,not arbitrary shader qualification.', '''
opaque_cuda_preallocated_scratch_alias_and_validation_contract|Does the native CUDA boundary preserve caller storage contracts?|Reuse scratch filled with0,165 and255; require identical outputs/VJPs,analytic area gradients,and pre-write rejection of short scratch,aliases,flags,device mismatch and bad indices.
''')

_add('packaging/test_opaque_mesh_preflight','Conservative projected-box admission only. These checks neither compute visible regions nor certify nonlinear shading or dense-frame speed.', """
mesh_candidate_count_matches_inclusive_native_predicate|Can the preflight miss triangles that native admission would retain?|Compare the difference-array count at every pixel with an independent direct closed-box predicate.
mesh_candidate_count_preserves_exact_boundaries_and_one_ulp_offsets|Can rounding hide a triangle exactly on a pixel border?|Compare exact-boundary and adjacent binary64 cases against direct per-pixel comparisons.
mesh_candidate_cap_is_not_silently_truncated|Does preflight report an overflow honestly?|Distinguish32 and33 coincident candidate faces without clipping the count.
mesh_candidate_invalid_or_unbounded_profiles_refuse|Can malformed or oversized input bypass bounded inspection?|Reject invalid extents, nonfinite positions and out-of-range triangle indices before allocation.
""")

_add('packaging/test_opaque_loss_extension','Independent cubic residual algebra, not general geometry or horizon derivative admission.', """
loss_extension_polynomial_difference_encloses_independent_piecewise_values|Can crossing an angular-table knot be bounded without pretending it is smooth?|Evaluate separate90-digit Bernstein polynomials at both sides and knots, and require their difference to lie inside the interval polynomial residual.
loss_extension_same_branch_cancels_exactly_and_refuses_moving_angles|Does the reference avoid invented residual error and unsupported derivatives?|Require exact zero when the actual and extended segment coincide, and explicit refusal of angular tangents and the horizon region.
centered_loss_preserves_constant_under_wide_high_mixed_jets|Can the checker invent variation in a constant matte curve?|Evaluate a constant curve over a broad cosine interval through order10 and first parameter motion using both interval backends. The centered polynomial must retain the exact constant and zero derivatives without dropping terms.
centered_loss_matches_independent_cubic_and_first_motion|Does the tighter matte bound still describe exactly the same curve and gradient?|Compare the log-cosine cubic against independent closed coefficients through order10, including first parameter motion and points outside the central segment. This checks the algebraic enclosure, not a renderer image or complete-piece gradient.
centered_horizon_extension_retains_linear_join_and_exact_branch_residual|Does the bound repair preserve the horizon segment and residual model?|Keep the linear horizon branch and its motion unchanged; require exact zero residual on the selected smooth branch. No knot or positive contribution is removed.
""")
_add('python/test_opaque_loss_extension_material','Finite independent material identity at roughness1; no dense-scene or geometry-gradient certificate.', """
material_extension_plus_residual_contains_independent_value_and_albedo_derivative|Does the residual repair the full Principled expression, including its material derivative?|Force adjacent smooth segments, add the separately enclosed diffuse residual, and require containment of the independent90-digit radiance and affine albedo derivative.
""")
_add('packaging/test_opaque_shading_intervals','Smooth quadrature plus separately enclosed residual; derivative and resource limits remain explicit.', """
interval_quadrature_adds_independent_signed_correction_and_tangent|Are signed residuals and their first derivatives integrated with the correct area?|Compare a constant signed correction and tangent with exact triangle integrals.
interval_quadrature_bounds_a_nonsmooth_residual_without_differentiating_it|Can a knot residual be bounded without applying a false smoothness assumption?|Enclose a positive-part quadratic with a known integral and keep the zero-residual negative control distinct.
""")

_add('packaging/test_opaque_shading_intervals','Residual enclosure and exact integration measure; material tangents only when angular transport is unsupported.', """
interval_correction_includes_exact_moving_area_derivative|Does residual integration include a changing triangle area?|Compare the value, material tangent and moving-area term with exact rational formulas.
interval_residual_freezes_one_smooth_extension_per_cell|Can adaptive branch choices silently alter the integral being certified?|Freeze one smooth extension for every cell, including all quadrature nodes and residual bounds; require both branches and the exact positive-part integral.
""")

_add('packaging/test_opaque_loss_extension','Smooth coordinate-range bound only; no sampling-based integration certificate.', """
cosine_mean_value_bounds_preserve_geometry_dependency|Can interval dependency force unnecessary angular branches?|Intersect the natural range with a centre-plus-enclosed-derivatives range; recover exact affine cancellation and contain independent point values of a varying nonlinear field.
""")

_add('packaging/test_opaque_mesh_evidence','Saved original pixel at(160,128) in the256-square camera. One albedo tangent,not geometry gradients or full-mesh acceptance.', """
saved_original_pixel_matches_reference_and_device_bytes|Does the installed renderer match the independent original-mesh pixel bounds?|Verify producer hashes and limits,exact full-pixel area,RGB and albedo-tangent bounds; check CPU/CUDA bytes separately. Reuses saved evidence without rerendering.
saved_pixel_comparison_rejects_wrong_values_and_wide_reference|Can a permissive evidence reader hide incorrect results?|Require refusal of wrong pixel values and reference intervals wider than the existing1e-12 radius limit.
""")

_add('packaging/test_opaque_knot_transport','Independent finite transport algebra; ordinary branch derivatives alone cannot certify a discontinuous moving integral.','\npiecewise_residual_first_derivative_contains_independent_polynomials|Are both sides of an angular knot differentiated correctly?|Compare complete ordinary residual derivatives with independent90-digit polynomials on both sides and at the knot.\ntransversal_flux_bounds_a_moving_discontinuous_step|Can pointwise differentiation miss an entire gradient?|Use a moving jump whose ordinary derivative is zero and whose exact integral derivative is5/6; require the separate transport bound to contain it.\ntransversal_flux_refuses_stationary_crossing_and_ignores_absent_knots|Is a nontransversal crossing silently certified?|Refuse when neither spatial partial is bounded away from zero; require zero flux for absent knots.\nresidual_derivative_refuses_horizon_and_spatial_jet_misuse|Can unsupported smoothness assumptions enter the reference?|Reject horizons and high spatial derivative jets at the piecewise residual boundary.\n')

_add('python/test_opaque_knot_material','One actual roughness1 angular knot with fixed F0; not arbitrary roughness or all moving geometry.','\nmoving_view_across_resource_knot_has_bounded_pixel_derivative|Does a changing view retain its angular-table boundary contribution?|Enclose the smooth integral,ordinary piecewise correction and knot-jump flux independently; compare a90-digit boundary identity and installed material integration.\n')

_add('python/test_opaque_knot_device','One fixed-visibility pixel under rigid projected motion. CPU/CUDA conformance is separate from independent accuracy and exact-byte equality.','\nmoving_geometry_knot_pixel_matches_independent_transport|Does moving geometry transport its attached view field across a knot?|Use installed geometry quadrature and full Principled material; compare RGB and the opposite-sign motion derivative with the saved independent enclosure on each device.\n')

_add('python/test_opaque_horizon_transport','Finite roughness1,F0=1,albedo0 with an explicit linear horizon. General production horizon partitioning is still unqualified.','\npartitioned_horizon_pixel_matches_closed_integral_and_boundary_flux|Can a horizon produce the wrong gradient sign?|Compare native unpartitioned and explicitly partitioned pixels with independent moving-region intervals and90-digit antiderivatives. The view negative control must have the wrong sign; both corrected view/light cases must satisfy unchanged tolerances.\n')

_add('packaging/test_opaque_frame_store','Development internal-disk storage,not a production rendering API or1080p rendering-speed claim. Existing memory and time limits remain unchanged.','\nframe_store_exact_ordered_roundtrip_and_no_overwrite|Are output and cotangent bytes preserved exactly?|Round-trip signed zero,subnormal and ordinary binary64 bytes in canonical pixel order; refuse an existing destination.\nframe_store_rejects_incomplete_reordered_and_oversized_writes|Can incomplete or reordered frame data be marked complete?|Reject out-of-order and oversized blocks before writes; refuse to seal a missing-pixel frame.\nframe_store_pins_identity_and_checks_bytes_before_replay|Can corrupted or mismatched evidence reach replay?|Require the supervising manifest hash and identity; reject changed payload before exposing its block and reject invalid file length.\nframe_store_full_hd_storage_and_ordered_cotangent_replay|Can full1080p data be stored without retaining full-frame tensors?|Write and verify132710400 bytes of synthetic output/cotangent data with at most32768 payload bytes per block; verify exact ordered replay. This measures storage,not rendering.\n')

_add('python/test_opaque_frame_replay','Finite3x3 installed scene with texture and varying roughness. Equality is per device against its resident baseline; arbitrary frame accuracy and throughput remain open.','\nframe_store_preserves_scene_pixels_and_ordered_vertex_vjp|Does disk-backed replay alter real renderer pixels or gradients?|Persist installed regional RGB,mask and cotangents,then replay pixels in canonical order; require exact equality with resident RGB,mask and vertex VJP on CPU or CUDA.\n')

_add('python/test_opaque_frame_cost','A passing gate test records a refused rendering request. No dense worker or frame-speed estimate is produced.','\noriginal_opaque_startup_capacity_and_missing_frame_anchors|Can old or partial timings qualify the new opaque renderer?|Inspect original32/64-square startup capacity,record pixels exceeding32 candidates,and require refusal when two matching complete-frame anchors are missing.\n')


_add('python/test_opaque_scene_horizons','Finite linear production horizons with conservative refusal of unresolved quadratic and grazing boundaries; unchanged independent accuracy limits.', """
production_view_horizon_matches_independent_pixel_and_motion|Does production rendering integrate a moving view horizon correctly?|Compare the actual scene colour and first normal-motion derivative to independent moving-partition interval integration; check coverage with the existing1e-12 area criterion.
production_curved_horizon_refuses_even_equal_vertex_normal_values|Can equal vertex samples hide an unqualified curved horizon?|Give independent differentiable vertex normals equal current values and require refusal rather than treating their field as constant.
production_light_horizon_cut_and_constant_view_motion_are_connected|Do moving directional-light boundaries remain connected to gradients?|Inspect the production homogeneous cut and require its boundary derivative to retain normal motion.
production_horizon_cut_capacity_is_shared_and_grazing_refuses|Can horizon handling silently raise work limits?|Require the shared16-cut cap and explicit refusal of grazing cases.
""")
_add('python/test_opaque_exact_diagnostic','Exact byte checks are separate from numerical tolerance; captures bind the installed wheel and preserve earlier failures.', """
capture_weighted_scene_gradient_byte_difference|At which stage do CPU and CUDA begin to disagree?|Capture nine global pixels, projection, quadrature, interpolants, shading, weights and their adjoints on CPU and two CUDA repeats; require exact final bytes in qualification mode.
projection_stages_are_device_exact|Can scalar image-size division perturb a camera projection?|Compare actual camera projection, inverse, combined matrix, clip and NDC values on CPU and CUDA at a3x3 off-centre camera.
""")
_add('python/test_opaque_knot_exact','Finite caller arithmetic diagnosis using unchanged installed native/material operations and saved independent bounds.', """
knot_device_exactness_requires_explicit_caller_reductions|Is the tiny knot derivative discrepancy in the renderer or in the caller expression?|Retain ordinary framework reductions as a control, then use explicit channel, sample and broadcast sums; compare per-batch images, vertex VJPs and motion derivatives byte for byte and against independent intervals.
""")
_add('python/test_opaque_window','Public bounded global-camera windows; a window is not a claim that a complete dense frame is admitted.', """
public_windows_equal_resident_pixels_and_canonical_vjp|Does external window storage change images or gradient accumulation?|Compare every3x3 window pixel and canonical row-major vertex VJP to the resident public renderer through the lazy facade.
public_window_validates_extent_before_allocation_and_input_mutation|Can invalid rectangles or changed inputs reach backward?|Reject invalid bounds/types/budgets and detect in-place changes between forward and replay.
public_window_full_frame_reserve_is_replaced_by_explicit_window_reserve|Can a host request a bounded region of a larger camera without retaining its entire image?|Preserve the resident1080p reserve refusal while admitting one explicit global-coordinate pixel; no1080p render timing is inferred.
""")
_add('python/test_opaque_frame_execution','Complete finite scene execution through installed public windows and host-owned internal storage; distinct from original-mesh or1080p qualification.', """
frame_executor_refuses_mutation_cancellation_and_corrupt_storage|Can cancelled, changed or corrupted execution be treated as complete?|Require cancellation without sealing, input-version refusal and payload-hash refusal before gradient replay.
complete_frame_execution_with_ordered_storage_and_vjp|Does an entire admitted finite frame survive bounded storage and backward replay exactly?|Render every pixel, seal complete ordered files, replay every cotangent in canonical order and compare all RGB/mask/vertex-gradient bytes with the resident public renderer; preserve device memory and process limits.
""")


_add('python/test_opaque_curved_preflight','Original128-square scene geometry only; no render or shaded accuracy admission. Earlier32-piece refusals remain visible.', """
original_curved_horizon_localization|Locate a real curved view boundary on the original mesh.|Classify exact rational Bernstein signs, preserve over-cap pixels, and isolate a visible crossing using depth-ordered independent fragments. Approximate roots select a probe only.
""")
_add('packaging/test_opaque_curved_reference','Development-only conic integration, with unchanged outward error and work limits. Production curved horizons still refuse.', """
parabolic_coverage_and_boundary_motion_have_closed_integrals|Keep the derivative caused by a moving curved boundary.|Compare coverage and motion against a90-digit parabolic antiderivative. A fixed-boundary pointwise derivative is zero and must fail this criterion.
complementary_regions_and_linear_limit_preserve_partition_and_vjp|Preserve complete area and gradients on both sides of the curve.|Integrate complementary regions, require their areas sum to the triangle and their boundary derivatives cancel, and verify the linear limit.
moving_triangle_jacobian_and_horizon_transport_both_survive|Include geometry motion as well as horizon motion.|Scale an entire triangle and its boundary; require the exact area derivative from the moving Jacobian.
uncertified_topology_and_work_limits_are_refused|Refuse unsupported topology rather than silently approximate it.|Reject tangencies, vertex crossings, uncertified radial derivatives and work budgets outside the unchanged profile.
perspective_quadratic_matches_independent_rational_corner_algebra|Construct the correct quadratic from perspective attributes.|Compare the expanded homogeneous polynomial to independent rational corner interpolation at distinct points and parameter values.
radial_field_composition_keeps_integral_and_parameter_derivative|Avoid thin-triangle interval cancellation without changing the model.|Compose fields exactly into radial coordinates and require a separate closed linear-field moment and its boundary derivative.
""")
_add('packaging/test_opaque_curved_original','One actual original visible triangle with a finite analytical Principled control material; not the complete original material,whole pixel,production rendering or frame qualification.', """
original_visible_conic_patch_value_and_first_normal_motion|Independently bound a real curved-horizon radiance contribution and its motion.|Pin the saved original geometry, certify its conic region, and integrate roughness=F0=1,albedo=0 radiance plus front area with separate1e-12-radius value and normal-motion bounds. Preserve the earlier harness and deadline failures.
""")

_add('python/test_native_opaque_curved','Installed shared C11 CPU/CUDA conic quadrature. Strict topology and the existing32-piece/800-node limits remain; shading accuracy is qualified separately.', """
curved_area_boundary_flux_and_complement|Does the curved boundary preserve coverage and its motion?|Compare native front area and parameter derivatives with an80-digit parabolic antiderivative; both sides must retain the exact total area and cancel their boundary flux.
curved_all_geometry_and_coefficient_gradients|Differentiate every curved input coordinate.|Gradcheck all vertex coordinates and all six conic coefficients,including sample motion and the curved area Jacobian.
curved_refuses_tangent_uncertified_and_combined_capacity|Refuse unsupported topology and storage before accepting a result.|Use an exact dyadic vertex event,uncertified interior curve,nonfinite inputs and the shared16-cut capacity. Preserve the earlier decimal-event harness failure.
curved_shared_kernel_bytes_and_vjps|Does CUDA reproduce the shared CPU arithmetic exactly?|Compare raw bytes of owners,samples and complete vertex/coefficient VJPs in float32 and float64; numerical closeness cannot substitute for this test.
""")
_add('python/test_opaque_curved_pixels','Finite roughness=F0=1,albedo0 controls. Original full-material RGB and the final mesh showcase remain open.', """
original_complete_visibility_and_curved_patch_matches_saved_reference|Apply native curved integration to the saved original-mesh patch.|Compare face5743 radiance,front area and normal-z motion to the preserved independent interval reference; separately check the complete depth-visible mask. Other faces RGB is not admitted by this control.
complete_production_curved_pixel|Validate every RGB channel and normal-motion gradient of a complete production pixel.|Compare the public scene renderer to an independent curved integral with explicit per-corner normalization. Keep the failed one-strip gradient and both harness errors; no tolerance was increased.
""")
_add('packaging/test_opaque_curved_pixel_reference','Outward integration of the declared rounded packed-input model; existing normalization acceptance remains separate.', """
complete_curved_pixel_reference|Build a complete-pixel reference without running the renderer.|Compose corner normalization and its first derivative independently at90-digit precision,then certify the curved region and enclose RGB and motion within the unchanged1e-12 radius and work caps.
""")
_add('python/test_opaque_curved_frames','Complete finite curved-scene execution,not original10000-face or1080p qualification. Failed frame gradients remain historical evidence.', """
complete_curved_frame_storage_replay_and_integrated_reference|Can a complete curved frame retain both accuracy and bounded replay?|Write every pixel and cotangent through public windows to internal host storage,replay the canonical first VJP,and compare resident bytes plus the frame integral with independent reference bounds. Enforce existing resource and cost gates.
""")
_add('python/test_opaque_scene_horizons','Finite production horizon controls; equal samples do not remove independently varying normal degrees of freedom.', """
production_equal_vertex_normals_keep_quadratic_motion|Do equal current vertex normals retain their independent motion?|Require an explicit quadratic input,then compare complete RGB and the collective normal-z derivative against the independent linear-limit integral.
""")

_add('python/test_opaque_full_original','Actual original material on all visible pieces. Execution/device checks do not replace independent integration-error bounds.', """
extract_complete_original_material|Which exact original faces and material are being tested?|Verify the corpus and saved projected geometry,then preserve all eight candidates,13 exact depth-visible pieces,packed normal/view fields and the frozen angular-loss resource.
complete_original_material_pixel|Can the complete original pixel and albedo gradients execute within existing limits?|Use the installed public window API with F0=.04 and blue RGB albedo;save exact CPU/CUDA bytes and CUDA peak memory. The original memory and capacity refusals remain history. Independent full-material bounds are still required.
constant_material_reservation_retains_unknown_cut_fallback|Does the lower reservation follow from the actual scene shape?|Constant materials have no texture or material-transition cut columns. Reserve every possible camera/light column and retain all other geometry charges; unknown attribute profiles retain the full576 slots per face.
""")
_add('packaging/test_opaque_square_reference','Independent outward reference controls,not renderer agreement or full-original accuracy.', """
square_polynomial_and_first_tangent|Integrate a known polynomial and its parameter derivative.|Compare the outward enclosure with exact rational moments;preserve the initial missing-implementation and Decimal expected-value harness failures.
square_nonlinear_closed_integral_and_tangent|Check a nonlinear integral without another renderer.|Compare reciprocal-field integrals and first derivatives against90-digit closed logarithmic formulas.
square_curved_area_includes_boundary_flux|Keep the curved boundary contribution in direct-square integration.|Compare a parabolic region and its moving-boundary flux with independent closed formulas.
square_refuses_unresolved_and_resource_growth|Do reference failures remain visible?|Refuse singular domains and requests beyond256 cells;depth and total outward-error limits remain enforced.
horizon_extension_encloses_first_linear_and_log_segments|Does the angular residual include the near-horizon linear segment?|Enclose differences against independently evaluated branch polynomials,including the first join;reject unsupported moving-angle tangents.
""")
_add('packaging/test_opaque_full_material_reference','Required full-material accuracy gate. A failed enclosure is not a passing renderer test.', """
full_original_material_reference|Independently bound the original pixel across all visible pieces.|Use exact saved visibility,curved horizon maps,and the actual frozen Principled loss resource. RGB and albedo derivatives require1e-12 reference radius within90 seconds and256 cells. Retain all harness,depth and cell-budget refusals. No original-frame or final showcase acceptance follows from CPU/CUDA agreement alone.
""")

_add('packaging/test_opaque_peano','Independent fixed-geometry error bounds; closed controls do not qualify the complete original pixel or moving-angle gradients.', """
peano_encloses_step_ramp_and_curvature_jumps|Can a quadrature error bound retain discontinuities?|Compare signed Gauss residual integration plus explicit value,slope and curvature jumps with exact rational integrals for five polynomial degrees.
peano_tensor_scaled_box_and_tiny_residual|Are physical box sizes included in every error term?|Integrate a diagonal clipped quadratic over an offset rectangle and compare its exact rational moment without changing1e-12 acceptance.
peano_negative_bounds_and_unqualified_tangent_refuse|Can an invalid proof contract silently pass?|Reject negative error bounds and parameter tangents that lack moving-boundary transport.
piecewise_log_residual_encloses_derivatives_and_binary_knot_jumps|Are angular joins assumed to be perfectly smooth?|Check ordinary derivatives through third order and a deliberate tiny endpoint jump against independent log-polynomial formulas.
square_integrator_uses_peano_correction_without_relaxing_tolerance|Does the new correction preserve the existing acceptance rule?|Require the complete summed outward enclosure to contain the exact clipped-quadratic integral within1e-12 radius.
radial_cosines_exact_polynomials_preserve_value_and_spatial_derivatives|Does algebraic reduction change the normalized material directions?|Compare exact rational dot-product reduction with independent normalized-vector formulas and their first three spatial derivatives.
peano_kernel_hulls_are_certified_not_sampled_maxima|Are the tighter constants bounds or sampled estimates?|Use exact de Casteljau subdivision and Bernstein convex hulls of the error-kernel polynomials;retain both supremum and absolute-integral bounds.
reference_stop_is_not_retried_as_a_subdivision_domain|Does the checker respect a stop immediately?|A supplied deadline must propagate after one correction call;preserve the regression where it was incorrectly retried as a domain failure.
""")
_add('packaging/test_opaque_full_material_reference','Diagnostic only: fixed saved boxes do not establish complete-pixel accuracy.', """
original_residual_error_components|Which part of the error bound is consuming the reference budget?|Record ordinary third-derivative and value,slope,curvature-jump bounds on three original-material boxes;do not treat this diagnostic as an integral pass.
""")
_add('python/test_opaque_full_original','Only one independently certified original visible piece;the complete13-piece pixel and frame remain unqualified.', """
first_original_material_piece_matches_independent_certificate|Does real native shading agree with the first certified original piece?|Pin the saved partial certificate for face5739;compare installed CPU/CUDA RGB and the albedo Jacobian against its outward intervals under unchanged tolerances. Keep the failed complete-pixel attempt visible.
""")

_add('packaging/test_opaque_peano','Existing exact-reference complexity limit; no expanded arithmetic allowance.', """
radial_cosines_retains_exact_rational_complexity_limit|Does polynomial reduction preserve the rational arithmetic cap?|Construct bounded inputs whose dot product exceeds8192bits and require refusal instead of admitting an unbounded intermediate.
""")

_add('packaging/test_opaque_branch_partition','Independent fixed-geometry bounds only; no complete original-pixel, moving-angle or frame admission.', """
affine_surrogate_exact_polynomial_and_curvature_bound|Does the angular classifier enclose its nonlinear field?|Compare a quadratic field against exact rational points and its outward Hessian remainder.
exact_affine_partition_preserves_area_and_tiny_slivers|Can partitioning lose a thin visible region?|Split an exact rational triangle by three lines and preserve total area and every positive sliver.
partition_zero_constant_plane_has_one_owner|Does an identically zero classifier duplicate coverage?|Require one polygon owner for constant classifiers and exact unchanged area.
partition_limits_and_cancellation_are_enforced|Does the reference stop before unbounded work?|Reject extra planes and excessive rational size, and propagate a caller cancellation immediately.
mismatch_error_retains_tiny_jump_and_quadratic_term|Do stored join discontinuities remain in the error bound?|Bound a known tiny value jump plus quadratic branch difference without assuming perfect continuity.
affine_enclosure_rejects_unqualified_parameter_tangents|Can a fixed-geometry proof silently qualify moving geometry?|Reject a field carrying a parameter tangent before constructing its classifier.
simplex_cosines_match_independent_normalization_formula|Does changing triangle coordinates preserve the material directions?|Compare simplex polynomial reduction and three spatial derivatives with independently normalized vectors.
join_bound_uses_outward_sub_ulp_interval_endpoints|Can rounding erase a tiny interval around a knot?|Use a nondecimal rational knot and radius below ordinary Decimal precision; require outward endpoints.
bounded_partition_refinement_preserves_area_error_and_depth|Do subdivisions retain the shared proof allowance?|Check exact area, summed quadratic error, inherited face identity and bounded depth/cell visits.
partition_refinement_propagates_deadlines_and_depth_refusals|Are terminal failures retried as more subdivisions?|Propagate the deadline after one call and refuse the unchanged depth limit.
displaced_branch_integral_encloses_closed_nonlinear_boundary|Does the mismatch bound enclose an actual integrated curved strip?|Integrate x+epsilon*y^2 crossing a linear classifier exactly, including value and slope jumps; compare with rational band area times the certified branch difference.
""")
_add('packaging/test_opaque_full_material_reference','Diagnostic completion is not numerical acceptance; all work refusals remain visible.', """
original_angular_partition_diagnostic|How many angular joins cross each original visible piece?|Inspect all thirteen saved pieces and report classifier errors or explicit knot-cap refusals without evaluating an integral.
partition_minimum_work_diagnostic|Can this partition strategy fit even before quadrature refinement?|Charge every classifier visit and one root per nonblack subregion, substituting no numerical field evaluations; an exhausted budget rules out an unchanged full retry.
""")
_add('packaging/test_opaque_full_material_reference','Whole original pixel requires the existing independent radius and shared resource bounds. First-piece progress alone is not a pass.', """
partitioned_original_material_reference|Does certified angular partitioning qualify all original shaded pieces?|Integrate selected smooth branches with outward Gauss bounds and add the displaced-boundary enclosure; preserve the knot-cap and256-cell failures without resetting budgets per piece.
""")
_add('python/test_image_saved_recovery','Current saved-parameter replay; historical optimization is not relabelled as a new solve or unique parameter identification.', """
saved_nine_parameter_solution_retains_current_image_acceptance|Does the preserved fit still satisfy its independent objective on the current wheel?|Pin historical report,JUnit and recipe hashes; evaluate initial and recovered parameters twice against fresh scalar targets, preserve locked inputs and require original one-percent-MSE acceptance.
""")

_add('packaging/test_opaque_peano','Independent fixed-geometry controls only; the original whole-pixel budget failures remain failures.', """
residual_direction_guides_refinement_without_relaxing_error|Does refinement follow the coordinate that actually contributes error?|Integrate the exactly known one-axis clipped quadratic in both orientations;enclose8/81 at1e-12radius within64cells without subdividing the constant coordinate. Preserve both red cap failures.
peano_directional_components_retain_complete_error_enclosure|Can the work-order hint replace the error certificate?|Check a closed two-axis cubic integral and the known seven-to-one directional error ratio;require the complete summed outward interval. Preserve the earlier incorrect positive-lower-bound harness failures.
""")
_add('packaging/test_opaque_square_reference','The aggregate reference acceptance and Decimal60 precision remain unchanged; controls do not qualify the original full pixel.', """
piece_error_budget_is_area_weighted_and_conserves_whole_pixel_radius|Can unequal piece sizes share one error allowance safely?|Allocate downward-rounded shares from exact rational areas,including zero measure;prove their sum remains at most5e-13 per basis channel and finalRGB radius at most1e-12.
outward_products_match_exact_endpoints_for_all_signs_and_ambient_precision|Does faster interval multiplication keep outward rounding?|Compare225 interval products across positive,negative and crossing-zero domains to exact rational endpoint extrema under deliberately low ambient Decimal precision.
""")

_add('packaging/test_opaque_reference_campaign','Contract controls validate evidence handling,not the original numerical integral. The approved campaign stopped on its second piece.', """
complete_campaign_composes_exact_integrals_and_unchanged_radius|Can independent piece certificates be assembled without losing the error guarantee?|Use thirteen analytically integrated constant triangles,exact rational answers and outward sums;check originalRGB and constant-albedo bounds.
campaign_rejects_missing_mismatched_or_modified_evidence|Can missing pieces or mismatched evidence silently pass?|First require the unmodified fixture to pass,then change ordering,input,material,table,geometry,source,tolerance,intervals or hashes and require refusal.
campaign_rejects_failed_or_overbudget_jobs|Can a failed or over-budget job count as accepted?|Begin with a valid campaign;inject invalid counts,times,memory,exit status or supervisor failure and require rejection even after rehashing.
campaign_prefix_stops_at_first_failure_without_accepting_partial_pixel|Does first-piece progress get mistaken for full accuracy?|Accept only an ordered successful prefix;refuse incomplete composition and a failed later result.
campaign_prospectively_reserves_remaining_aggregate_work|Are limits checked before starting more work?|Reserve a complete next job from the unchanged aggregate allowance;reject attempts after all thirteen jobs.
campaign_loader_checks_original_file_bytes_before_parsing|Do stored certificates remain tied to their exact bytes?|Check SHA256 before parsing,reject a whitespace-modified file and refuse overwriting an existing result.
piece_selector_retains_original_all_piece_error_allocation|Does the entry point reject unsupported piece selections?|Reject negative,out-of-range,boolean and noninteger indices before preparing the expensive reference.
selected_piece_uses_its_original_share_and_no_other_piece|Does a selected piece keep the full pixel's original error share?|Mock the integrator only to inspect routing:select first and last original pieces,require exactly one call and their original all-piece tolerance. Mock values are never accuracy evidence.
""")
_add('packaging/test_opaque_reference_piece','One independently bounded fixed-geometry piece. All thirteen must pass before whole-pixel acceptance;moving gradients remain separate.', """
original_piece_independent_certificate|Can this original visible piece be certified within its approved allowance?|Pin mesh,material,angular table,source hashes and original area-weighted tolerance;run outward integration under256cells/depth10/90seconds and a monitored1GiB worker. Preserve a failure and stop the campaign immediately.
""")

_add('packaging/test_opaque_localized_peano','Independent analytical reference controls only. No original-pixel,moving-gradient,frame or final-showcase pass is implied.', """
localized_jump_encloses_exact_truncated_power|Can knowing where a jump lies reduce uncertainty without losing the true integral?|Use exact step,ramp and curvature integrals at both cell boundaries and the middle;bound every overlapping kernel polynomial and compare the full enclosure.
oblique_crossing_retains_value_slope_and_curvature_jumps|Does a slanted crossing preserve all discontinuity terms?|Integrate a truncated quadratic with nonzero value and slope jumps exactly over a square;compare localized and global bounds at the same amplitude.
crossing_location_uses_certified_mean_value_inclusion|Can a root position be narrowed safely for every transverse coordinate?|Check increasing and decreasing affine fields,interval-valued knots,no-root exclusion and full-domain fallback when the derivative contains zero.
local_kernel_hull_covers_gauss_node_uncertainty_and_global_fallback|Can a kernel jump at a quadrature node be accidentally omitted?|Include both neighboring polynomial pieces around uncertain nodes and retain the valid whole-cell bound as a fallback.
localized_witness_refuses_invalid_domain_or_jump_bounds|Can malformed crossing evidence narrow an error bound?|Reject positions outside the physical cell,negative jump bounds and incomplete jump vectors.
localized_witness_count_remains_bounded|Does the new path retain finite internal work?|Reject more than sixteen localized witnesses per channel,matching two angular roles with eight knots each.
material_producer_localizes_real_log_spline_jumps_and_preserves_integral|Does the material checker actually use the certified locations correctly?|Construct a piecewise-log loss with explicit value/slope jumps,derive its integral by a separate closed antiderivative and retain both ordinary and localized uncertainty.
material_nonmonotone_roots_keep_existing_count_and_curved_refusal|Do ambiguous derivative signs lead to an unsafe single-root assumption?|Keep the complete domain and four-root view bound for a noncurved example;continue refusing a curved nonmonotone profile.
work_refusal_preserves_last_complete_enclosure_and_error_components|Will a failed refinement leave useful whole-domain evidence?|Stop an exact x-to-the-tenth integral at one cell and preserve the last full cover;also distinguish ordinary and localized material correction errors. Neither becomes a pass.
""")

_add('packaging/test_opaque_signed_peano','Fixed-geometry independent analytical controls. Signed enclosures are not original-mesh or moving-gradient qualification.', """
signed_jump_recovers_exact_truncated_power|Can the checker retain a boundary correction's direction?|Use exact step,ramp and quadratic integrals with increasing and decreasing crossings; subtract the signed quadrature error and retain the true answer.
oblique_and_curved_boundary_enclose_exact_polynomial|Does a boundary that varies across a pixel keep a conservative bound?|Compare slanted and quadratic crossing curves to exact rational double integrals,including ordinary and optional-crossing uncertainty.
optional_crossing_includes_zero_error|What if the crossing is absent on some transverse lines?|Keep zero in the contribution interval whenever presence is not certified for every line.
signed_kernel_covers_both_sides_of_uncertain_nodes|Can signed bounds lose a discontinuity at a Gauss node?|Enclose both neighboring kernel branches;this caught an ambient-precision Decimal negation defect,now repaired with exact sign reversal.
signed_witness_requires_single_crossing_and_valid_intervals|Can ambiguous crossing evidence create a signed certificate?|Reject multiple crossings,non-Boolean presence and signed values outside their declared absolute bounds.
signed_physical_lengths_and_transverse_mass|Are signed corrections scaled correctly outside the unit square?|Use closed step,ramp and quadratic integrals on an offset rectangle with unequal side lengths.
""")
_add('packaging/test_opaque_localized_peano','Finite material-checker qualification only; the original campaigns remain stopped.', """
material_signed_crossing_has_correct_orientation_and_log_integral|Does the actual material producer preserve the direction of its jumps?|Check both coordinate orientations against a closed piecewise-log antiderivative. The initial whole-bound2x prediction failed because ordinary error dominates;retain that failure and measure signed boundary uncertainty separately.
material_partial_crossing_keeps_absence_and_nonmonotone_fallback|Can optional or nonmonotone crossings become falsely certain?|Require endpoint-certified presence for signed terms;retain zero for optional crossings and the old multiplicity bound for nonmonotone cases.
signed_material_refinement_reduces_work_with_same_acceptance|Does the tighter material bound reduce work without relaxing acceptance?|Compare signed and unsigned refinement at the same1e-12 radius,256cells and depth10;both enclose the exact log integral,with3 versus7cells on each named fixture.
""")


_add('packaging/test_opaque_transverse_peano','Independent fixed-geometry reference controls. No original-mesh, moving-gradient, complete-frame or final-showcase acceptance follows.', """
kernel_primitive_preserves_exact_zero_moment|Does the integrated error kernel retain its exact zero moment?|Integrate each of the three Gauss error kernels across the full normalized interval and enclose zero with outward arithmetic.
integrated_kernel_matches_independent_polynomial_antiderivative|Is boundary integration checked against a separate derivation?|Compare the primitive with directly integrated active truncated powers on a clipped interval.
clamped_root_encloses_exact_affine_crossing|What happens when a crossing leaves the cell?|Check roots before,inside and after a nonunit physical interval in both orientations;retain clipped endpoints.
clamped_root_refuses_nonmonotone_map|Can an ambiguous root become a certified coordinate change?|Refuse derivative intervals containing zero.
weighted_kernel_keeps_coefficient_variation_and_endpoint_uncertainty|Can cancellation erase uncertainty in the material term?|Keep both endpoint uncertainty and a certified absolute-mass bound for coefficient variation;check independent admissible extreme cases.
conditioned_error_recovers_diagonal_clipped_polynomial|Does the tensor error identity preserve the exact double integral?|Check closed step,ramp and quadratic integrals with both orientations;retain value-jump ambiguity at Gauss nodes.
conditioned_integral_rejects_tangents|Can spatial integration silently qualify moving geometry?|Reject parameter tangents lacking separate boundary-transport evidence.
material_transverse_producer_encloses_clipped_log_rectangle|Does the actual material producer enclose oblique angular joins?|Compare twelve clipped and reversed log-spline cases with a separate closed double antiderivative;include stored value,slope and curvature jumps.
material_transverse_curved_map_preserves_jacobian_and_exact_integral|Does a curved coordinate change preserve measure?|Include the nonlinear map Jacobian and compare the transformed log-material integral to a closed rectangle integral.
material_transverse_zero_derivative_keeps_conservative_fallback|What if the transverse angle does not vary?|Keep the existing whole-domain bound when the change of variable is inadmissible.
material_transverse_work_uses_same_accuracy_and_cell_budget|Does the tighter bound pay for its extra per-cell work?|Measure both methods at the same1e-12radius and256cell/depth10 limits;new7cells/387input calls versus121/4043 and133/4439,with both enclosing the exact integrals.
material_transverse_nonunit_box_keeps_physical_scale|Are physical dimensions retained outside the unit square?|Integrate an offset rectangle with unequal sides and both crossing directions against the same exact log formula.
material_transverse_nonmonotone_outer_map_keeps_whole_domain_bound|Does an ambiguous transverse derivative lead to unsafe cancellation?|Retain the conservative witness when the outer derivative spans zero;curved nonmonotone evaluation still refuses.
transverse_integral_refuses_unsigned_witness|Can an unsupported signed correction enter the bound?|Reject an integrated contribution without its single-crossing signed witness.
""")


_add('packaging/test_opaque_boundary_diagnostic','Fixed-box diagnostic controls only; no original accuracy or full-frame acceptance.', """
diagnostic_refuses_invalid_scope_before_evaluation|Can a diagnostic silently become an unrestricted integration run?|Reject zero or more than three boxes,malformed dimensions,and coordinates outside the unit square before evaluation.
diagnostic_refuses_expanded_limits|Can the diagnostic bypass its prospective work allowance?|Reject expanded,nonfinite,zero and noninteger limits;at most2048correction input calls and20seconds.
diagnostic_counts_attempts_stops_before_extra_input_and_restores|Is work charged when the diagnostic stops?|Count the first input,refuse the next at the cap,and restore the original callbacks without retrying.
diagnostic_deadline_cancellation_restores_without_retries|Does interruption leave the material checker usable?|Exercise deadline and explicit cancellation,propagating the stop and restoring callbacks.
kernel_parts_reconstruct_outward_enclosure|Do component diagnostics describe the actual bound?|Retain the exact anchored primitive interval and coefficient-variation radius for all three kernel orders.
diagnostic_reports_two_orientations_without_integrating_field|Are both tensor directions inspected without claiming a complete integral?|Count every witness and material input,report both error decompositions,and forbid adaptive integration.
diagnostic_exposes_fallback_reason|Can the report distinguish failed coordinate-change admission?|Use a zero outer derivative and record the conservative fallback reason.
""")
_add('packaging/test_opaque_boundary_original_diagnostic','Three overlapping saved boxes from failed original piece1 only; not a full-piece decomposition or certificate.', """
saved_original_boundary_boxes|What makes the original material error bound wide?|Verify frozen input and failed-trace hashes,inspect cells30/31/32 in both directions under20seconds/2048correction inputs,and expose22witnesses with root,coefficient and uniform-bound accounting. The initial interval-construction fixture failure is preserved.
""")
_add('packaging/test_opaque_endpoint_peano','Independent analytical endpoint controls. These do not certify the original mesh or moving gradients.', """
endpoint_mass_contains_point_primitives_and_never_widens|Can uncertain primitive endpoints be bounded without losing cancellation?|Check all three kernels across twelve endpoint cases,including a Gauss-node crossing,against point primitives and the previous valid enclosure;require at least twofold tightening.
uncertain_endpoints_in_first_segment_enclose_exact_polynomial|Does the tightened enclosure retain an independent closed answer?|Before the first Gauss node,K3=t^3/6;enclose the exact fourth-power endpoint expression and retain the original width requirement. A first implementation failed that tightness check;certified monotonicity repaired it.
signed_witness_intersects_two_valid_bounds_and_refuses_disjoint|Can integrating a witness make its certified bound worse?|Intersect the integrated and uniform enclosures;retain the tighter valid interval and refuse contradictory disjoint certificates.
""")


_add('packaging/test_opaque_smooth_peano','Bounded fixed-geometry reference controls only. No complete original-pixel, moving-gradient, frame or showcase acceptance.', """
constant_third_derivative_has_zero_centered_remainder|Can an exactly constant third derivative cancel safely?|Use cubic exactness of Gauss5 and signed constant derivatives; require a zero ordinary remainder while boundary terms remain separate.
centering_retains_full_signed_range_and_physical_scale|Does centering retain all derivative uncertainty?|Keep the complete signed range and fourth-power inner length times outer measure; compare the resulting outward bound.
smooth_bound_refuses_inconsistent_derivative_evidence|Can contradictory derivative evidence narrow the bound?|Reject negative absolute bounds and signed ranges extending beyond their declared magnitude.
smooth_bound_preserves_unsigned_fallback|What if no signed derivative range is available?|Retain the prior ordinary bound exactly and record the absence of a constant anchor.
material_producer_keeps_signed_third_without_extra_inputs|Does reporting a signed derivative create hidden evaluations?|Count the actual material inputs and validate the signed range against the existing absolute third bound.
diagnostic_components_reconstruct_both_tensor_radii|Do reported components reconstruct the actual error interval?|Sum ordinary and all boundary radii in both tensor directions, allowing only outward interval rounding.
fixed_subboxes_keep_all_derivative_work_and_exact_cubic|Is the four-region improvement bounded and correctly charged?|Count exactly four evaluations on nonunit boxes in both directions and signs; constant third derivatives must cancel.
partitioned_bound_retains_change_between_inner_halves|Can separate anchors erase a real change across the pixel?|Use different derivatives in the inner halves and require nonzero uncertainty; one shared inner anchor is essential.
partitioned_bound_refuses_bad_scope_and_propagates_stop|Can the new bound bypass cancellation or expand its scope?|Reject an invalid axis before input evaluation and propagate cancellation before any derivative work.
partitioned_remainder_encloses_exact_truncated_polynomial|Does the tighter bound still enclose independent exact integrals?|Compare truncated cubic, quartic and sextic integrals in both directions, signs and physical scales against rational antiderivatives.
partition_diagnostic_charges_eight_extra_calls_and_keeps_base_result|Does a prospective diagnostic alter the baseline result?|Count four additional derivative evaluations per direction while preserving both original tensor-error intervals.
partitioned_material_retains_exact_log_integral_and_all_jumps|Does the actual material path retain angular discontinuities?|Use twelve clipped and reversed piecewise-log cases with a closed double antiderivative; retain value, slope and curvature jumps.
partitioned_material_curved_map_retains_exact_measure|Does the bound remain valid under a curved coordinate map?|Include the nonlinear Jacobian and compare the resulting integral against the independent log rectangle formula.
partitioned_material_work_is_charged_at_unchanged_accuracy|What does the tighter bound cost at the same acceptance?|Retain the 1e-12 radius and cell/depth caps; measure calls, cells and elapsed time for baseline and candidate against exact log integrals.
integrated_ordinary_bound_never_replaces_conditioned_mass|Can an integrated witness incorrectly replace a conditioned remainder?|Allow the extra bound only for the integrated term, preserve the conditioned term and reject negative evidence.
""")
_add('packaging/test_opaque_square_reference','Bounded evidence accounting only; a stopped cover does not become an accuracy pass.', """
complete_cover_accounting_replaces_parent_and_retains_failure|Do complete-cover summaries replace parents with retained children?|Check exact component totals after subdivision and preserve the last complete cover when a cell cap stops the next refinement.
complete_cover_accounting_caps_leaf_details|Can diagnostic accounting grow without limit or invent missing evidence?|Retain at most five largest leaves, reject more than256 entries and report missing components as unknown.
""")

_add('packaging/test_opaque_boundary_original_diagnostic','Three overlapping saved D325 boxes only. No complete piece or pixel, renderer accuracy, moving gradient or showcase claim.', """
saved_original_smooth_components|How much ordinary remainder can four derivative regions safely remove?|Verify original input and D325 failure hashes, inspect cells30/31/32 in both directions under20seconds/2048calls, retain18boundary witnesses and charge every extra derivative evaluation. Compare the connected candidate with the preserved baseline.
""")


_add('packaging/test_opaque_conditioned_center','Bounded fixed-geometry reference controls. No whole original-piece, moving-gradient, frame or final-showcase acceptance.', """
conditioned_constant_third_cancels_but_keeps_jumps|Can the smooth remainder cancel without erasing material discontinuities?|At each fixed outer Gauss node use one anchor across the whole inner interval; check both tensor directions and signs, retaining all boundary jumps.
conditioned_centered_bound_encloses_exact_truncated_polynomial|Does the centered bound still enclose a known integral?|Compare independently integrated truncated polynomial controls in both directions with their exact rational answers.
centered_error_refuses_inconsistent_signed_evidence|Can inconsistent derivative evidence narrow a certificate?|Refuse a signed range that contradicts the existing absolute derivative bound.
local_kernel_reuse_is_bounded_and_preserves_all_signed_ranges|Does reuse preserve the exact kernel bounds and memory cap?|Compare cached and direct outward ranges at exact Decimal endpoints, require a cache hit, and enforce a maximum of128entries.
cached_ranges_do_not_expose_mutable_entries|Can one caller corrupt a later cached enclosure?|Mutate a returned interval and require the next lookup to return independent, unchanged endpoints.
conditioned_center_retains_exact_material_and_boundaries|Does the material producer preserve stored angular jumps?|Check twelve clipped and reversed piecewise-log fixtures against a separate closed double antiderivative, with value,slope and curvature discontinuities retained.
center_and_kernel_reuse_preserve_exact_acceptance_and_charge_work|Does the repair reduce work at unchanged accuracy?|Compare two log-material cases with the same1e-12radius and resource bounds. Record cells,input calls,kernel builds and time; independently enclose the exact integral. Earlier acceptance need not produce a narrower final interval.
""")
_add('packaging/test_opaque_boundary_original_diagnostic','Three disjoint largest retained leaves from D327 only; no whole-piece certificate or prediction.', """
saved_largest_leaf_terms_and_directions|Which error terms and integration direction matter on the saved original regions?|Pin input and failure hashes, inspect the three largest retained leaves in both directions under20seconds and2048input calls, and compare ordinary and boundary terms without adaptive integration.
""")


_add('packaging/test_opaque_material_reuse','Bounded fixed-geometry reference qualification only. No whole original-pixel, moving-gradient, frame or final-showcase acceptance.', """
axis_bounds_does_not_repeat_identical_input_derivatives|Does the checker recompute an identical derivative unnecessarily?|Require one input-jet evaluation for a constant-angle axis bound while retaining the existing material formula.
absolute_kernel_bounds_share_existing_signed_cache|Can absolute and signed error bounds share the same calculation?|Require an existing exact-key cache hit and identical absolute endpoints at full,small,node-crossing and endpoint domains. Retain the128entry capacity.
jump_inputs_use_certified_region_and_preserve_spatial_jet|Are jump derivatives evaluated only where crossings can occur?|Check both axes,point and interval crossings against independent polynomial jets. Preserve the active spatial derivative and avoid reevaluation for unchanged regions.
localized_jump_encloses_exact_log_material|Does a narrower crossing region still enclose the true material integral?|Compare twelve clipped and reversed log-material fixtures against a separate exact double antiderivative,with all stored value,slope and curvature jumps retained.
jump_localization_propagates_stop_before_extra_inputs|Does the extra evaluation honor cancellation?|Require cancellation before any additional crossing-region input evaluation.
""")
_add('packaging/test_opaque_boundary_original_diagnostic','Three saved D329 regions only. Profiling changes timing; it is not production throughput or a whole-piece certificate.', """
saved_boundary_uncertainty_and_derivative_cost|What causes boundary uncertainty and reference cost on the original mesh?|Pin original input and failed-result bytes;inspect the three largest retained leaves in both directions under20seconds/2048inputs. Record root/variation terms and profiled function costs;inclusive times overlap. Compare exactly matched regions before and after the repair.
""")


_add('packaging/test_opaque_smooth_branch','Independent development-reference controls for fixed geometry only. These are enclosure tests and an algebraic interval argument, not an external proof certificate or whole original-pixel acceptance.', """
centered_branch_encloses_independent_log_polynomial_derivatives|Does the equivalent polynomial keep the correct derivatives?|Convert exact constant, linear, quadratic and cubic log polynomials to Bernstein controls. Compare values and three spatial Taylor coefficients against separate 100-digit derivative formulas at three cosines and both directions.
centered_branch_keeps_uncertain_controls_over_interval|Are uncertain input controls still enclosed?|Retain outward interval coefficients over a nonzero cosine interval. Independently check all 16 endpoint-control combinations at three points; finite samples support the test, while the algebraic identity and outward operations justify inclusion across the interval.
centered_branch_cache_is_bounded_invalidates_and_returns_fresh_jets|Can cached coefficients silently change the answer?|Mutate returned intervals and source controls independently. Require fresh results, changed-control invalidation and at most two stored branch tuples.
centered_branch_refuses_invalid_indices|Does the helper reject ambiguous branch requests?|Reject boolean, out-of-range and unsupported negative indices rather than choosing an unintended segment.
centered_branch_refuses_unqualified_order|Does the helper stay within the qualified derivative scope?|Refuse orders above three so the local correction qualification cannot silently change the separate high-order smooth-base checker.
centered_branch_keeps_horizon_formula_and_rejects_moving_angle|Are horizon behavior and moving-angle limits preserved?|Check the exact existing linear horizon values and slopes, and require a refusal for an unqualified moving-angle tangent.
""")


_add('packaging/test_opaque_third_piece_diagnostic','Bounded investigation of three pinned regions from the saved D332 piece-2 failure. Passing means the investigation completed within its limits, not that the original pixel or renderer is qualified.', """
saved_third_piece_terms_directions_and_cost|Which uncertainty term is holding up the next piece?|Inspect the same three largest saved regions in both integration directions, retaining ordinary and boundary terms. Limit the diagnostic to eight seconds and 1024 correction inputs; record preparation separately and preserve the source and failure identities.
saved_third_piece_conditioned_dependence|Would simply halving each line fix the loose bound?|Within twelve seconds and 1024 correction inputs, inspect two halves of each of thirty conditioned lines. Separate view, light and mixed derivative terms and count possible material segments. Record hypothetical bounds but return the original bounds unchanged. Saved-report comparison must establish exact equality of those returned bounds; this is not an adopted repair or a full trial.
""")


_add('packaging/test_opaque_segmented_remainder','Independent controls for the development checker. Exact real identities justify the enclosure; separate numerical controls exercise implementation and refusals. This is not a proof-assistant certificate, whole original-piece acceptance or a native renderer test.', """
segmented_remainder_preserves_nonzero_signed_cubic_mass|Does every material segment keep its signed contribution?|Compare the signed remainder against separate 100-digit Gauss algebra and the closed integral of a truncated cubic at four boundaries, traversed in both directions. Subinterval kernel masses must not be silently discarded.
uncertain_and_endpoint_roots_retain_every_piecewise_cubic|Does uncertainty about a boundary remain in the bound?|Cover the whole boundary interval, including boundaries touching either endpoint. Compare separate closed cubic results at the interval endpoints and midpoint; the complete interval enclosure supplies the continuum argument.
constant_derivative_cancels_over_complete_domain|Is cancellation applied only where it is valid?|Use a single full-domain anchor across several subintervals and require an exactly zero regular remainder for constant third derivatives, including a non-unit physical domain.
segmented_partition_is_bounded_and_stops_before_evaluation|Can the repair escape its work limits?|Require fallback before evaluating derivatives if more than sixteen slabs would be needed. A stop request must propagate before any extra derivative call.
segmented_remainder_refuses_invalid_domains|Can malformed roots omit part of the integral?|Reject zero-width integration domains and root intervals outside the physical domain.
signed_ordinary_enclosure_keeps_every_jump_and_physical_scale|Does the tighter ordinary bound preserve the rest of the error?|Keep all value, slope and curvature jump charges while inserting a signed ordinary remainder under both signs and a non-unit domain and outer mass.
segmented_material_encloses_independent_log_integral|Does the full material calculation still contain the exact answer?|Compare twelve clipped log-material rectangles with a separately integrated analytic expression, using every pair of axis directions. Require containment and no widening beyond outward-rounding allowance.
segmented_material_keeps_nonmonotone_fallback|What happens when a line can cross a boundary repeatedly?|Keep the existing four-crossing conservative witness and omit the optional signed ordinary refinement.
segment_selection_is_oriented_and_keeps_uncertain_root_zone|Are material branches selected on the correct side?|Check increasing and decreasing guaranteed crossings. Outside the certified root interval select only the proved side; inside it retain all candidate branches.
absent_crossing_cannot_select_a_material_branch|Can a merely possible boundary justify discarding a branch?|No. Retain every candidate when crossing presence is not guaranteed. The initial failure of this regression is preserved before its repair.
segmented_curved_material_retains_jacobian_and_exact_integral|Do curved coordinates preserve the physical integral?|Use a monotone quadratic coordinate map with its varying Jacobian in four direction combinations, and compare with an independent closed log-material integral.
signed_ordinary_refuses_disjoint_evidence|Can contradictory bounds silently look more accurate?|Reject signed evidence that does not overlap the existing conservative ordinary bound.
full_original_producer_selects_qualified_segmented_checker|Is the qualified method connected to the real reference producer?|Load the pinned original input and intercept checker construction before any whole-piece quadrature. Require the new option and both existing integration options; preserve the test-first missing-binding failure.
""")

_add('packaging/test_opaque_third_piece_diagnostic','Pinned local investigation only; no whole-piece certificate.', """
saved_third_piece_segmented_remainder|How much does the qualified repair help the saved failing regions?|Inspect the same three largest saved piece-2 regions in both directions under twenty seconds and 2048 correction inputs. Compare their correction widths with the frozen baseline and preserve all integrated and jump contributions. Local timing does not establish complete-piece speed.
""")


_add('packaging/test_opaque_piece1_cost','A bounded saved-region investigation, not a complete original-piece certificate or runtime benchmark.', """
saved_piece1_matched_segment_cost|Where is the tighter bound spending its time?|Compare the three largest saved D335 second-piece regions in both directions. Pin input and failure hashes, retain every error term, charge at most twenty seconds and 2048 correction inputs across baseline and repaired runs, and record preparation and cache-state limitations.
""")
_add('packaging/test_opaque_input_reuse','Exact local reuse controls for the independent development checker. No renderer, moving-gradient or complete-frame qualification.', """
exact_polynomial_jets_and_returned_objects_are_independent|Can reused derivatives be changed by a caller?|Check exact polynomial jets in both directions, mutate returned objects, and require independent subsequent results without repeating the input calculation.
every_input_coefficient_and_order_is_part_of_identity|Could different derivative inputs share the wrong result?|Change bounds, derivative coefficients, order and an explicit zero. Require a new evaluation and exact agreement with the uncached calculation.
cache_is_bounded_and_new_call_scope_does_not_reuse_old_state|Can the cache grow or leak across geometry changes?|Limit storage to sixteen entries, evaluate excess keys without storage, and require a new cache scope to evaluate again.
cache_hit_still_checks_cancellation_and_failed_evaluation_is_not_cached|Can reuse bypass a stop or preserve a failed result?|Check cancellation before hits and require failed input evaluations to leave no cache entry.
reused_segmented_log_material_matches_uncached_and_exact_integral|Does reuse preserve the independently justified enclosure?|Test twelve log-material rectangles across all axis direction combinations. Require exact equality with the uncached interval and containment of the independently integrated answer.
""")


_add('packaging/test_opaque_kernel_cost','A local four-pass timing investigation. No whole-piece performance, renderer, gradient or M1 acceptance claim.', """
cold_cache_input_reuse_and_kernel_cost|Does exact reuse help when both paths start with empty caches?|Run uncached, reuse, reuse, uncached on the same three saved regions in both directions. Clear the same kernel and material caches before every pass, use identical timing instrumentation, preserve exact numerical fields, and charge all work within twenty seconds and 2048 correction inputs. Attribute kernel misses and repeated geometry keys without performing extra numerical trials.
""")


_add('packaging/test_opaque_requested_kernel','Independent checker controls; no complete original-mesh image or M1 acceptance claim.', """
requested_order_matches_full_outward_polynomial|Does requesting one order preserve its exact enclosure?|Compare both endpoints against the full-range interface for all three orders on endpoints, wide, narrow and uncertain domains.
requested_kernel_contains_independent_100digit_peano_value|Does the enclosure contain an independent mathematical value?|Evaluate the closed Peano sum with independent100-digit Gauss nodes and weights at seven points for each order; require containment.
requested_order_keeps_both_sides_of_uncertain_gauss_nodes|Are jump uncertainties preserved?|Test every uncertain Gauss node and all orders, retaining both sides and the first-order jump width.
requested_only_evaluates_one_order_and_uses_shared_bounded_cache|Does selective work retain bounded storage and safe returned values?|Observe only the requested order, mutate a returned interval, and exceed128positions to verify eviction and fresh return objects.
requested_order_refuses_invalid_order|Are invalid order requests rejected?|Require refusal of out-of-range orders, booleans, nonintegers and an absent order.
requested_order_refuses_invalid_domain|Are out-of-domain positions rejected?|Require refusal when a position extends outside the unit interval.
weighted_path_requests_only_its_order|Does the weighted integrator request exactly what it needs?|Observe every local-range call for each weighted order and require that same order.
requested_orders_share_one_position_entry_and_fill_only_missing_orders|Do different orders share one position entry?|Request the third order then all orders, checking that only the two absent orders are filled and subsequent reads do no new work.
failed_order_fill_keeps_other_completed_orders|Does interrupted work preserve previously completed bounds?|Force a fill failure, retain the already computed third-order result, and recover the missing first-order slot.
""")
_add('packaging/test_opaque_requested_kernel_cost','Saved-region investigation with exact numerical comparison; no whole-piece speed guarantee.', """
cold_cache_requested_kernel_cost|Can selective kernel work preserve every saved numerical bound within the investigation allowance?|Compare the same three saved regions in both directions with empty caches. Count computed orders and cache positions separately, retain all endpoint fields, and charge prior passes toward the aggregate20second/2048input limit. The final continuation has one observation, not a throughput estimate.
""")


_add('packaging/test_opaque_interval_work','Independent outward arithmetic controls; no tolerance or precision relaxation.', """
sign_selected_products_match_all_four_directed_endpoints|Does less multiplication preserve outward endpoints?|Compare every sign class and tiny/large magnitudes against all four endpoint products under the original directed rounding.
strict_signs_select_only_extremal_products|Which endpoint products are mathematically necessary?|Require two products for strict-sign extrema or four when both intervals straddle zero; zero endpoints keep original tie ordering.
jet_recurrence_matches_full_two_axis_reference|Are all live derivative coefficients preserved?|Compare reciprocal and square-root coefficients with the complete recurrence through order10, including tangents as small as1e-80.
jet_recurrence_contains_independent_binomial_coefficients|Do the coefficients contain exact mathematical answers?|Check independent rational binomial coefficients for values and mixed tangents through order10.
fixed_geometry_does_not_evaluate_absent_tangent_recurrence|Does fixed geometry avoid computing identically zero tangent rows?|Count interval multiplication work with absent versus tiny live tangents; reciprocal reuse changes division counts but must retain the complete live tangent recurrence.
""")
_add('packaging/test_opaque_d338_diagnostic','Three saved regions, both directions, bounded work; no complete-piece admission.', """
saved_d338_third_piece_terms_directions_and_cost|Where do the latest third-piece bounds and costs come from?|Inspect the pinned failed cover in both directions and preserve all terms. Compare repaired arithmetic exactly while charging prior passes to the same20second/2048input allowance.
""")
_add('packaging/test_opaque_smooth_base','Independent high-order smooth-base controls; every boundary correction remains required.', """
smooth_base_contains_independent_high_order_log_coefficients|Does the centered curve enclose high-order derivatives?|Compare through order10 with independently convolved110digit log-polynomial series, both directions and several points.
high_order_base_retains_uncertain_controls_and_shared_two_entry_limit|Does the high-order form preserve uncertain material data and bounded storage?|Check every control-box vertex against independent series and require mutation isolation and the existing two-entry cache.
high_order_base_keeps_horizon_and_refuses_moving_angles|Does high-order support respect its qualification boundary?|Retain the exact linear horizon formula and reject unqualified moving-angle tangents.
original_base_and_correction_share_the_qualified_branch_cache|Does the original reference use the qualified representation without another cache?|Intercept the actual producer and verify both order10 base calls use the same bounded cache as correction work.
centered_nonlinear_base_integral_matches_independent_log_antiderivative|Does the full smooth integral match a closed-form answer?|Integrate a cubic of log of a reciprocal field and compare against its independent analytical antiderivative at the unchanged radius.
""")
_add('packaging/test_opaque_smooth_base_diagnostic','Local base-bound comparison only; the entire original pixel still needs fresh certificates.', """
saved_d338_smooth_base_bounds_and_cost|Does the new representation tighten the actual saved base bounds?|Compare Bernstein and centered forms on the same three frozen regions, retain overlap, count all field calls and failed work, and enforce the remaining aggregate diagnostic allowance.
""")

_add('packaging/test_opaque_split_response','Three saved regions only; all failed diagnostic work remains charged.', """
saved_d340_split_response|Does the current axis hint select the more useful subdivision?|Compare complete base-plus-correction integrals for both midpoint splits of three saved regions. Residual-only comparison is invalid when child base branches differ; its historical failure remains visible.
""")
_add('packaging/test_opaque_signed_partition','Independent exact integrals qualify signed bounds; all boundary jumps remain.', """
signed_partition_contains_exact_truncated_polynomial|Does signed remainder contain a known quadrature error?|Compare against closed truncated-polynomial integrals for both axes, signs, physical scales and degrees; retain exactly four derivative evaluations.
piecewise_constant_third_has_nonzero_exact_signed_half_mass|Are nonzero half-interval kernel contributions retained?|Use an exact truncated cubic whose third derivative changes at the midpoint; compare the actual quadrature error, including its sign.
constant_third_preserves_zero_and_four_calls|Does an exact cubic still have zero ordinary error?|Check constant third derivatives and unchanged four-box work.
integrated_signed_bound_keeps_sign_mass_and_jumps|Are physical scaling and boundary terms retained?|Require signed center, no second mass multiplication, full jumps and refusal of contradictory evidence.
material_producer_supplies_signed_integrated_bound|Is the qualified bound connected to the material checker?|Require the signed integrated contribution inside its existing conservative bound.
""")
_add('packaging/test_opaque_signed_partition_diagnostic','Same saved regions; no complete-piece certificate or speed guarantee.', """
signed_partition_same_saved_regions|How much does the signed repair tighten the saved bounds?|Measure correction widths on the same three parent regions and charge all prior diagnostic work to20seconds/2048calls.
""")
_add('packaging/test_opaque_refinement_guidance','Scheduling advice never supplies numerical acceptance evidence.', """
guided_nonlinear_integral_and_tangent_still_match_closed_form|Does guidance retain independent integral and tangent accuracy?|Compare against closed logarithmic formulas under the existing tolerance, counting two guidance calls per cell.
zero_or_tiny_guide_never_admits_unresolved_bound|Can misleadingly small guidance falsely pass a result?|Require the real enclosure to fail at the one-cell limit despite zero or tiny guidance.
invalid_guidance_falls_back_but_stop_propagates|Does unavailable guidance preserve safe execution?|Fall back on ordinary guidance errors while propagating cancellation.
guidance_does_not_change_exact_one_cell_evidence|Does a large guide contaminate the accepted value?|Require identical exact one-cell value and tangent bounds.
original_checker_supplies_separate_refinement_field|Does the original producer separate scheduling from acceptance?|Verify a separate guidance callable while centered smooth bounds own acceptance.
guidance_cost_on_same_saved_regions|What extra work does the guide add?|Measure six order10fields on the same three saved regions, including preparation and all prior diagnostic charges; do not infer a full-piece improvement.
""")

_add('packaging/test_opaque_denominator_reuse','Exact outward arithmetic and object-isolation controls; no precision relaxation.', """
quotient_matches_independent_directed_endpoint_operations|Do quotient endpoints retain their original directed meaning?|Calculate both reciprocal endpoints and all product extrema independently at60digits, including signs and extreme scales under low ambient precision.
recurrence_uses_one_denominator_without_coefficient_change|Can one reciprocal serve the complete recurrence?|Require one local division and exact coefficient equality to the full recurrence through order10, including1e-80live tangents.
existing_interval_operands_do_not_need_temporary_copies|Can arithmetic read operands without copying them?|Count only required result objects and prove operands remain unchanged and separate from the result.
zero_addition_keeps_its_fresh_return_contract|Does the zero shortcut preserve mutation isolation?|Mutate its returned interval and require the original operand to remain unchanged.
""")
_add('packaging/test_opaque_d343_diagnostic','Same three frozen regions, both directions, aggregate resource accounting.', """
saved_d343_third_piece_terms_directions_and_cost|Does reduced arithmetic preserve every actual saved bound?|Compare all numerical fields exactly and charge every earlier diagnostic pass against20seconds/2048calls. Profile timings do not establish whole-piece throughput.
""")
_add('packaging/test_opaque_denominator_cost','Controlled old-new-new-old comparison; two observations per path.', """
saved_d343_base_guidance_arithmetic_cost|Does exact arithmetic reuse help the high-order base and guide?|Load only pinned historical arithmetic methods, clear branch caches, time preparation and both high-order fields on the same three boxes, and require exact coefficients with complete aggregate diagnostic charges.
""")

_add('packaging/test_opaque_tensor_order','Both tensor identities require complete independent error bounds; unchanged accuracy and resource limits.', """
both_orders_enclose_independent_material_integrals|Do both integration orders contain the analytical answer?|Check independently integrated logarithmic materials, both crossing signs, curved Jacobians and clipped value/slope/curvature cases.
physical_axis_callbacks_and_nonunit_polynomial|Are coordinates and lengths still physical after swapping order?|Verify a rational polynomial integral over a nonunit rectangle and inspect all five conditioned callbacks and the integrated callback.
invalid_order_refused_before_field_evaluation|Can an invalid order accidentally execute work?|Reject Boolean, missing, out-of-range and noninteger selectors before any field call.
original_profile_uses_qualified_fourth_piece_order|Which original piece changes order?|Inspect all thirteen producer selections: only index3 uses integrated axis0; all others retain axis1.
saved_fourth_piece_order_enclosures_and_cost|Does the alternative tighten the saved difficult regions?|Compare full correction enclosures on the same three saved leaves in both orders, retaining overlap and charging every diagnostic input and second. This is not a complete-piece certificate.
""")
_add('packaging/test_opaque_d345_diagnostic','Pinned fourth-piece failure evidence; bounded investigation only.', """
saved_d345_fourth_piece_terms_directions_and_cost|Where does fourth-piece uncertainty come from?|Inspect both integration directions and all ordinary and boundary terms on three largest disjoint saved leaves. Retain all historical failure evidence; do not infer renderer error from an uncertified bound.
""")

_add('packaging/test_opaque_owned_jets','Exact arithmetic with explicit ownership and mutation-isolation checks.', """
arithmetic_does_not_recopy_owned_coefficient_dictionary|Can freshly calculated coefficients transfer ownership?|Observe redundant constructor copies for five operators and retain tiny live tangents.
result_mutation_cannot_change_inputs|Can a result corrupt its operands?|Mutate every returned coefficient across five operators and missing/shared/tiny coefficients; require unchanged input dictionaries.
external_coefficient_dictionary_still_copies|Are user-supplied coefficient dictionaries isolated?|Mutate and clear the supplied dictionary; the constructed jet must retain its original value.
addition_uses_one_context_and_directed_endpoints|Does reduced context work retain outward rounding?|Compare independent floor/ceiling endpoints under low ambient precision and count one local rounding context.
decimal_endpoint_construction_avoids_conversion|Can known Decimal endpoints avoid conversion?|Preserve signed-zero and extreme-scale endpoint tuples while observing no redundant conversion.
decimal_fast_path_retains_invalid_endpoint_refusal|Does the faster constructor still reject invalid intervals?|Reject nonfinite and reversed Decimal endpoints.
saved_regions_retain_exact_bounds_with_owned_arithmetic|Does the real saved calculation retain exactly the same bound?|Compare full base-plus-correction endpoints, count every diagnostic input and second, and separate local timings from whole-piece qualification.
""")
_add('packaging/test_opaque_zero_material_work','Skip only structurally zero residual terms; every live coefficient is retained.', """
only_needed_smooth_branches_are_evaluated_with_exact_coefficients|Can zero residuals avoid irrelevant smooth work?|Compare every coefficient with the complete expression for all four live/zero combinations and both tested orders; count only needed smooth evaluations.
zero_correction_keeps_invalid_denominator_refusal|Can a zero result hide an invalid denominator?|Require the original denominator refusal even when both corrections are zero.
tiny_high_order_correction_is_not_treated_as_zero|Can a tiny derivative disappear in this shortcut?|Retain a nonzero order-three coefficient of1e-80 despite an exactly zero value coefficient.
""")
_add('packaging/test_opaque_d347_split','Bounded three-region investigation; no complete-piece acceptance.', """
saved_d347_split_response|Is the refinement scheduler splitting in the useful direction?|Inspect full base-plus-correction integrals for both midpoint splits of three saved leaves; preserve overlap and charge the complete diagnostic budget.
""")

_add('packaging/test_opaque_fused_products','A shared rounding context preserves separate directed products and sums; no relaxed acceptance.', """
fused_convolution_keeps_every_directed_endpoint|Does less arithmetic setup preserve the exact coefficients?|Compare endpoint tuples through order10 with the original separately rounded convolution, dense and sparse data, small live tangents and low ambient precision; mutate outputs to verify input isolation.
convolution_uses_one_local_rounding_context|Which overhead was actually removed?|Count one local rounding context per complete coefficient multiplication, retaining every product and addition.
coefficients_enclose_independent_rational_polynomial_product|Does the arithmetic enclose an independent mathematical answer?|Compare every retained coefficient with an exact rational polynomial convolution, including mixed tangents.
explicit_zero_coefficients_and_cancellation_preserve_zero_behavior|Are zeros and cancellation handled consistently?|Compare explicitly inserted zero coefficients and opposite-sign sums with the original endpoint behavior.
order_mismatch_and_caller_context_are_preserved|Can the optimization leak precision settings or mix incompatible jets?|Verify the caller precision, rounding and flags remain unchanged and mismatched orders are refused.
synthetic_material_expression_exact_bounds_and_controlled_timing|Does the same synthetic expression retain its bounds?|Run old/new/new/old paths with eight evaluations each; require exact coefficient tuples and report timings without claiming original-piece performance.
""")

_add('packaging/test_opaque_d351_diagnostic','Three pinned saved regions, both directions and all error components; no whole-piece admission.', """
saved_d351_fourth_piece_terms_directions_and_cost|What still controls fourth-piece uncertainty?|Inspect the latest three largest disjoint leaves in both tensor directions, preserving ordinary and boundary terms, input counts and instrumented costs under20seconds/2048inputs.
""")
_add('packaging/test_opaque_adaptive_order','A local direction score chooses work; only the complete selected interval can establish acceptance.', """
adaptive_material_keeps_independent_analytical_integrals|Does per-region selection retain independent material accuracy?|Check exact logarithmic integrals, crossing signs, curved Jacobians and physical scaling in ordinary and full segmented material configurations.
selector_charges_both_candidates_reuses_chosen_rows_and_contains_exact_polynomial|Is the extra selection work explicit and physically correct?|Require both continuous candidates, reuse only the selected rows, retain five opposite-axis lines and compare with an independent rational polynomial integral.
tie_keeps_preferred_axis_and_failure_of_alternate_keeps_valid_bound|Can unavailable alternatives invalidate an otherwise supported calculation?|Retain the preferred identity on ties or an unsupported alternative while still computing its full bound.
cancellation_and_two_unavailable_orders_cannot_be_accepted|Can fallback hide cancellation or absence of a valid proof?|Propagate cancellation and refuse when neither tensor identity is supported.
invalid_selector_is_refused_before_callbacks|Are invalid direction selectors rejected before numerical work?|Reject Boolean, missing, noninteger and out-of-range selectors before callbacks.
only_fourth_original_piece_enables_adaptive_order|Which original piece changes its strategy?|Inspect all13producer selections and require adaptive ordering only for the fourth piece.
same_saved_regions_adaptive_order_bounds_and_cost|Do tighter local bounds repay their extra computation?|Compare fixed and adaptive correction enclosures on the same three saved boxes; preserve overlap and charge every input and second. Local results do not establish a whole-piece win.
small_selection_score_never_replaces_full_conditioned_error|Can a promising selection score conceal a large remaining error?|Deliberately make the selected continuous score zero while the conditioned remainder is large; require the complete wide interval unchanged.
""")

_add('packaging/test_opaque_direct_kernel','Independent checker controls; numerical and resource acceptance unchanged.', """
direct_bernstein_contains_exact_rational_power_conversion|Do direct kernel coefficients enclose the exact polynomial?|Compare against independent rational power expansion and basis conversion.
uncertain_nodes_weights_and_endpoints_retain_all_corner_polynomials|Do uncertain inputs remain enclosed?|Check every corner polynomial of uncertain endpoints, weights and nodes.
local_kernel_no_longer_constructs_general_derivative_jets|Was generic derivative setup removed?|Forbid jet construction during local degree-three kernel evaluation.
dense_independent_kernel_points_and_jump_sides|Are actual Gauss kernels and both jump sides retained?|Use independent110digit values and uncertain node intervals.
""")

_add('packaging/test_opaque_kernel_mass','Independent checker controls; numerical and resource acceptance unchanged.', """
absolute_bernstein_mass_contains_exact_integrals|Is absolute polynomial area bounded?|Compare positive, negative and sign-changing Bernstein polynomials with exact integrals.
mass_is_bounded_by_existing_global_and_rectangle_bounds|Does the tighter mass retain prior safeguards?|Intersect certified local mass with existing global and rectangle bounds.
linear_kernel_mass_exact_in_first_segment_and_fresh_cache|Is mass reuse bounded and isolated?|Check exact linear area, fresh values and the128entry limit.
weighted_kernel_uses_integrated_mass_with_all_uncertain_support|Are variation bounds using complete support?|Retain all possible endpoints and independently known weighted area.
mass_does_not_rebuild_the_same_segment_polynomials_for_rectangle|Is duplicate polynomial work avoided?|Forbid a second range construction while computing the mass guard.
""")

_add('packaging/test_opaque_direct_kernel_cost','Independent checker controls; numerical and resource acceptance unchanged.', """
direct_kernel_same_regions_bounds_and_timing|Does direct algebra improve the saved-region cost?|Run old/new/new/old with20seconds2048inputs; preserve the failed ambient-precision assertion and all completed measurements.
saved_direct_kernel_comparison_uses_outward_margin|Can the saved diagnostic be audited without rerunning it?|Pin the report hash and check repeatability, interval overlap and an outward-rounded comparison margin.
mass_same_regions_with_remaining_diagnostic_allowance|Do tighter masses help the same regions?|Charge prior1560inputs and all preparation; compare three complete correction bounds without full-piece admission.
""")

_add('packaging/test_opaque_refined_roots','Root refinement and work order never replace full numerical acceptance.', """
quadratic_crossing_contracts_and_encloses_all_transverse_roots|Does another mean-value inclusion retain every root?|Check exact quadratic roots in both directions with transverse uncertainty and exactly one additional midpoint input.
root_refinement_cancellation_cannot_be_hidden|Can root refinement hide cancellation?|Propagate the stop rather than manufacture a crossing witness.
invalid_refinement_switch_refused|Are invalid switches refused?|Require an explicit Boolean before accepting material inputs.
refined_roots_keep_independent_material_integrals|Do narrowed root intervals preserve independent integrals?|Compare analytical logarithmic, curved and nonunit material integrals in both directions and complete material modes.
fourth_piece_uses_certified_width_without_extra_heuristic|Which original piece omits heuristic work?|Require the fourth piece to use certified-width priority and the other12to keep guidance; keep optional root refinement disabled.
certified_width_priority_keeps_closed_integral_and_refuses_cell_exhaustion|Does a simpler work order still enforce accuracy?|Check a nonlinear closed integral and tangent and reject a one-cell budget that cannot prove them.
""")
_add('packaging/test_opaque_refined_roots_cost','Pinned three-region comparison, not whole-piece qualification.', """
saved_root_contraction_enclosures_and_cost|Does root contraction repay its cost?|Use old/new/new/old, retain all enclosures and charge every preparation input within20seconds2048inputs; local benefit was insufficient for adoption.
""")

_add('packaging/test_opaque_knot_lookup','Live lookup reuse preserves every numerical comparison and endpoint.', """
segment_table_is_resolved_once_per_call_and_not_retained|Can we remove repeated table access safely?|Count one proxy lookup for16,128and512segments, preserve zero correction, and observe a replacement table on the next call.
segment_enclosures_keep_all_legacy_coefficients|Are angular bounds unchanged?|Compare every coefficient endpoint across overlapping knots and spatial orders0,1and3.
""")
_add('packaging/test_opaque_timing_variance','Three pinned regions under a shared2048input20second allowance; no whole-piece admission.', """
second_piece_same_regions_old_new_timing|Why did a previously passing piece time out?|Compare old/current methods on identical saved regions, record CPU and wall times separately and isolate the profiled sample; the full slowdown was not reproduced.
lookup_same_regions_with_remaining_budget|Does fewer table lookups reduce local cost?|Charge prior work, require exact saved numerical hashes and retain the post-measurement empty-profiler-summary failure.
completed_lookup_diagnostic_is_auditable_without_rerun|Can completed measurements survive a reporting failure?|Pin original report bytes and verify all six region results and aggregate budgets without repeating expensive work.
""")

_add('packaging/test_opaque_kernel_power_cache','Immutable finite polynomial constants; every requested bound remains outward and complete.', """
kernel_power_constants_reused_across_domains_and_safe_to_mutate_results|What can be reused without changing the kernel?|Reuse coefficients with full node/weight identity, return fresh intervals and reject stale reuse after a weight changes.
kernel_power_cache_has_fixed_eighteen_entry_bound|Can the cache grow indefinitely?|Verify its18entry maximum under40distinct inputs; existing worker memory cap remains authoritative.
cached_power_coefficients_contain_independent_exact_moments|Do cached coefficients represent the correct polynomial?|Check each coefficient against an independent exact-rational moment expansion for all three orders.
""")
_add('packaging/test_opaque_kernel_power_cost','Three pinned saved regions; local bounds and costs, not whole-piece admission.', """
saved_kernel_constants_enclosures_and_cost|Does polynomial reuse help the difficult regions?|Compare old/new/new/old complete corrections under2048inputs20seconds, retain all enclosures and report timing drift without promising full-trial success.
""")
_add('packaging/test_opaque_refined_roots','Historical heuristic restored after the cheaper schedule failed; complete bounds still own acceptance.', """
original_piece_guidance_and_optional_root_refinement_binding|Which strategy is active after the failed scheduling experiment?|Current D364: use certified-width priority for curved pieces, historical guidance for noncurved pieces, and keep optional root refinement disabled. Earlier runs tested the then-active strategy and retain their original outcomes.
""")

_add('packaging/test_opaque_conserved_allocation','Local shares may change; the total5e-13basis and final1e-12pixel uncertainty limits do not.', """
work_weights_conserve_exact_basis_and_final_rgb_radius|Does redistribution keep the original accuracy guarantee?|Use exact rational area/work sums through32pieces and extreme valid weights; prove the unchanged total and worst-caseRGB/derivative bound.
invalid_work_weights_cannot_erase_nonzero_pieces_or_expand_domain|Can a bad weight hide work or uncertainty?|Reject zero, negative, inexact, malformed and out-of-range weights before allocation.
zero_area_stays_exact_zero_with_positive_work_weight|Do degenerate triangles remain harmless?|Keep zero-area shares exactly zero while positive areas retain positive shares and the total remains bounded.
original_profile_redistributes_without_increasing_the_pool|Where does the difficult piece's extra share come from?|Require fourth weight5/2,others1, with all twelve other shares tightening after normalization; reject a non13piece profile.
campaign_rejects_extra_pool_even_if_every_receipt_is_rehashed|Can consistent hashes conceal a larger error allowance?|Double one share without compensation and require the unchanged aggregate verifier to reject it despite fresh matching hashes.
campaign_spec_uses_the_same_explicit_original_work_profile|Do producer and verifier agree on each share?|Compare the frozen spec allocation with the exact declared work profile.
nonzero_width_certificates_compose_at_unchanged_final_bound|Do uncertain piece results still prove complete accuracy?|Compose independently known constant integrals with nonzero outward radii and enforce the original finalRGB and derivative bound.
""")

_add('packaging/test_opaque_domain_direction','A bounded diagnosis does not certify the whole pixel.', """
saved_ninth_domain_directions|Why did the ninth piece reach the depth limit?|Inspect three saved regions, both angle roles and both subdivision directions; count all input evaluations including preparation and preserve domain refusals.
""")
_add('packaging/test_opaque_horizon_factor','Exact conic algebra removes cancellation before rounding; all original checks remain.', """
root_factor_preserves_point_jets_and_exact_zero|Can we evaluate a horizon without losing its exact zero?|Compare both conic sides and derivative orders0,1,3,10 with independent90digit values and the direct identity.
radial_derivatives_overlap_direct_identity|Are radial derivatives preserved?|Retain all derivatives through order10 under the original fourth-power distance map.
mismatched_horizon_and_missing_distance_refuse|Can the wrong horizon certify a value?|Reject any conic whose exact rational numerator differs from the packed normal/view dot product, or missing distance evidence.
""")
_add('packaging/test_opaque_domain_split','Scheduling chooses work; complete outward enclosures decide acceptance.', """
direction_hint_keeps_closed_integral_with_fixed_depth|Does choosing the useful split direction preserve accuracy?|Recover an independent exact integral under a fixed depth cap; still refuse insufficient depth.
invalid_hints_keep_fallback_and_cannot_admit|Can a malformed hint bypass a failure?|Reject invalid hints as scheduling evidence and retain the original domain failure and fallback split.
cosine_direction_evidence_reuses_existing_evaluations|Does directional guidance add hidden kernel work?|Obtain both directions from exactly the four existing mean-value field evaluations.
range_fallback_retains_complete_residual_and_domain_refusals|Can a nonsmooth residual still have a rigorous bound?|Use the complete range times exact area, retain all physical contributions, and refuse the unchanged segment cap.
material_range_fallback_encloses_independent_closed_integral|Does range fallback preserve real material contributions?|Check four orientations and a varying Jacobian against independent closed logarithmic integrals.
""")

_add('packaging/test_opaque_domain_direction2','Inspect a saved failure without admitting it.', """
saved_ninth_shifted_domain_directions|Does a smaller region meet the existing angular guard?|Measure both directions and roles on three saved boxes with charged preparation and strict diagnostic limits.
""")
_add('packaging/test_opaque_domain_direction3','A local bound is evidence about a region, not the whole piece.', """
saved_ninth_power_domain_directions|Which part of the curved bound remains loose?|Inspect the saved eighth-power failure and both coordinate halves, preserving all refusals and cost.
""")
_add('packaging/test_opaque_horizon_power','A coordinate substitution must preserve the surface and its full Jacobian.', """
flattened_map_has_independent_closed_area_and_boundary_motion|Does flattening keep the exact area and moving boundary?|Compare fourth/eighth powers on both conic sides against closed area and derivative formulas.
flattened_map_preserves_point_coordinate_jacobian_and_factored_cosine|Are coordinates and derivatives preserved?|Check order10 jets, the full Jacobian and exact horizon zero under both qualified maps.
unqualified_powers_are_refused|Can an arbitrary mapping silently enter qualification?|Reject unsupported or malformed powers before evaluation.
original_ninth_piece_bounded_preflight|Can the repaired checker certify the difficult ninth piece?|Run one frozen diagnostic at original accuracy and worker limits; preserve the complete bound or exact refusal. This is never reused as a campaign certificate.
""")
_add('packaging/test_opaque_monotone_bounds','Only whole-rectangle derivative signs certify extrema.', """
certified_corner_ranges_contain_exact_rational_interior|Can corner bounds safely tighten angular ranges?|Test all sign orientations against121independent exact rational interior values each.
unknown_sign_keeps_whole_domain_and_does_not_sample_corners|What if monotonicity is unknown?|Keep the whole-domain range and perform no unjustified corner tightening.
shared_corners_are_evaluated_once_and_cancellation_propagates|Is added work bounded and interruptible?|Reuse common corners and propagate cancellation without inventing a bound.
""")
_add('packaging/test_opaque_residual_range','Complete polynomial envelopes retain discontinuities and all material factors.', """
range_index_contains_independent_discontinuous_polynomials|Can one envelope include many distinct branches?|Check complete and partial ranges against independent90digit discontinuous polynomials, both channels and different base branches.
horizon_zero_and_snapshot_are_retained|Are horizon and frozen-resource semantics preserved?|Keep exact zero inside the linear horizon branch, preserve uncertain endpoints and isolate the prepared snapshot from later mutation.
invalid_resource_and_cancellation_are_refused|Can bad ordering or cancellation be ignored?|Reject unordered resources and stop preprocessing immediately when requested.
whole_residual_range_encloses_closed_material_integral|Does the alternative proof retain real material contributions?|Compare four orientations and a varying Jacobian with independent closed logarithmic integrals; retain only the scheduling hint, never a shortcut to acceptance.
missing_top_domain_is_refused_instead_of_omitted|Can a truncated table silently omit high angles?|Reject a resource that does not cover the exact unit-cosine endpoint.
""")
_add('packaging/test_opaque_residual_range_cost','Preparation is included in bounded diagnostic cost.', """
saved_ninth_whole_residual_envelopes|How costly and tight is the alternative residual bound?|Build the full frozen range index and evaluate three saved regions, counting input work and recording complete outward widths.
""")

_add('packaging/test_opaque_aggregate_peano','Independent fixed-geometry proof controls; no whole-pixel or moving-gradient admission.', """
aggregate_jump_sums_cover_independent_log_derivatives|Are every knot and jump order included?|Compare summed value, slope and curvature jumps against independently differentiated log polynomials.
aggregate_material_bound_contains_closed_log_integral|Does the aggregate proof enclose the correct integral?|Check four coordinate orientations and a varying Jacobian against a closed logarithmic rectangle integral.
more_than_eight_discontinuous_steps_keep_every_jump|Can a broad range lose discontinuities?|Integrate sixteen exact step branches and retain every value jump.
unknown_monotonicity_and_cancellation_refuse|Are crossing assumptions enforced?|Reject unknown directional monotonicity and propagate explicit cancellation.
all_ordinary_jet_orders_enclose_independent_formulas|Do ordinary derivatives remain branchwise valid?|Compare all orders through three with independent90digit formulas, retaining every branch.
simultaneous_view_and_light_steps_preserve_product_jumps|Can coincident angular boundaries hide a product term?|Compare both simultaneous step roles against the exact weighted piecewise integral.
empty_knot_range_has_exact_zero_jump_sum|Does an empty query invent uncertainty?|Require an exact zero sum when no knot is present, without dropping any nonzero jump.
aggregate_adoption_intersects_bounds_and_preserves_failures|Does the production fallback retain independent evidence?|Intersect complete ranges; preserve monotonicity fallback and reject cancellation or disjoint bounds.
conditioned_aggregate_encloses_independent_material_integral|Can the two integration orders be interchanged safely?|Check both tensor identities against closed integrals in four orientations.
""")
_add('packaging/test_opaque_aggregate_cost','Saved-region measurement includes proof preparation; it is not a whole-piece certificate.', """
saved_complete_regions_aggregate_bounds_and_cost|How tight and costly is the aggregate proof?|Compare three preserved regions with original complete bounds under20seconds and2048inputs.
""")
_add('packaging/test_opaque_aggregate_diagnostic','Separate smooth, ordinary and knot terms; local improvements cannot admit the original pixel.', """
saved_ninth_smooth_and_residual_directions|Which terms still prevent acceptance?|Inspect three saved leaves, both smooth directions and both conditioned correction orders with bounded work.
""")
_add('packaging/test_opaque_aggregate_transport','Transport requires certified signs over the entire region; all lower jump orders remain.', """
full_sweep_bound_retains_curvature_and_lower_jump_terms|When is complete-boundary cancellation valid?|Test the exact zero kernel moment in all four direction orientations while retaining outward uncertainty and lower jumps.
unknown_outer_sign_keeps_uniform_bound|What if boundary transport cannot be proved?|Retain the conservative uniform jump bound when an outer derivative contains zero.
transported_integral_contains_independent_log_squared_rectangle|Does transported evidence contain the actual integral?|Compare both tensor orders and all direction signs against an independent90digit closed log-squared integral.
no_knots_do_not_require_inner_monotonicity|Does a nonexistent crossing need a direction proof?|Preserve exact zero jump work when the angle stays in one branch.
invalid_integrated_bound_is_rejected|Can invalid jump evidence enter acceptance?|Reject a negative supplied integrated bound before using it.
""")
GUIDES['tests/packaging/test_opaque_aggregate_transport.py::collection'] = dict(purpose='Could this historical test module be collected?',method='Display the original collection error and immutable report hash; no test body ran.',limits='Historical development failure, not a numerical execution or acceptance result.')

_add('packaging/test_opaque_grouped_peano', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
grouped_curvature_encloses_closed_log_squared_integral|Does signed grouping preserve the actual integral?|Compare both integration orders and four orientations with an independent90digit closed log-squared integral.
grouped_retains_discontinuities_and_product_jumps|Can grouping erase discontinuities?|Retain16step branches, simultaneous angular crossings and closed logarithmic integrals.
disjoint_signed_jump_evidence_is_rejected|Can incompatible evidence be accepted?|Supply a signed jump interval disjoint from its conservative bound and require refusal.
grouped_cancellation_and_unknown_monotonicity_refuse|Are crossing assumptions and cancellation enforced?|Refuse uncertain directional signs and propagate cancellation.
""")

_add('packaging/test_opaque_high_peano', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
all_kernel_orders_enclose_independent_truncated_power_error|Are all ten error kernels enclosed?|Compare interval kernels with independently evaluated90digit truncated-power moments.
high_branch_derivatives_enclose_independent_recurrence|Are high derivatives valid within each branch?|Compare with an independently derived logarithmic-polynomial derivative recurrence.
old_low_order_guard_remains_and_high_order_is_bounded|Does the new helper weaken the old guard?|Keep the order3interface strict and cap the separate high-order helper at10.
complete_high_identity_contains_exact_truncated_polynomial_integral|Are every derivative jump and ordinary term retained?|Compare orders2through10 with exact truncated-polynomial integrals through degree12.
""")

_add('packaging/test_opaque_high_material', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
high_material_contains_closed_log_squared_integral|Does high-order material integration preserve the integral?|Compare five derivative orders, both axes and four orientations with independent closed integrals.
high_material_retains_product_jumps_and_refusals|Can higher order omit discontinuities or refusal?|Retain sixteen steps, simultaneous product jumps, closed logarithmic integrals and invalid monotonicity/cancellation controls.
""")

_add('packaging/test_opaque_high_transport', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
signed_kernel_primitive_contains_independent_truncated_moments|Is the signed boundary moment correct?|Compare full, partial and zero supports with independent90digit exact moment formulas through order6.
transport_keeps_independent_log_squared_rectangle|Does following boundaries preserve the actual integral?|Compare both axes and four direction orientations with an independent closed log-squared rectangle integral.
transport_rejects_disjoint_integrated_evidence|Can incompatible transported evidence be accepted?|Require refusal when a supplied integrated jump contradicts the conservative interval.
transport_preserves_discontinuous_and_refusal_controls|Are discontinuities and stopping rules preserved?|Keep all sixteen steps, simultaneous crossings, uncertain-sign refusals and cancellation.
""")

_add('packaging/test_opaque_grouped_diagnostic', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
saved_ninth_grouped_terms_and_directions|Does signed grouping help the saved difficult regions?|Measure both tensor orders on three preserved regions, including preparation, within20seconds and2048inputs.
""")

_add('packaging/test_opaque_high_diagnostic', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
saved_ninth_high_order_terms_and_directions|Does order10 improve difficult regions within the budget?|Measure three saved regions under20seconds/2048inputs; retain the deadline failure and partial payload, without treating it as a certificate.
""")

_add('packaging/test_opaque_order_diagnostic', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
saved_ninth_lower_order_terms_and_directions|Are cheaper lower orders more useful?|Compare selected orders2through4 on the same three saved regions under unchanged diagnostic caps.
""")

_add('packaging/test_opaque_term_diagnostic', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
saved_ninth_per_term_uncertainty|Which proof terms dominate the remaining uncertainty?|Separate ordinary derivatives and every jump on one saved region at orders3,4,6; whole and fixed transverse domains remain distinct.
""")

_add('packaging/test_opaque_transport_diagnostic', 'Experimental independent-reference evidence only; no original-pixel, moving-gradient or M1 admission.', """
saved_ninth_signed_transport|Does signed transport justify another trial?|Measure both tensor orders on three preserved regions under20seconds/2048inputs; local gains alone cannot qualify the piece.
""")

_add('packaging/test_opaque_boundary_taylor', 'Experimental independent-reference work only; limits and original accuracy acceptance remain unchanged.', """
circle_level_set_derivatives_match_closed_form|Are implicit boundary derivatives correct?|Compare both circle branches and three points with independent90digit closed derivatives through order3.
curve_box_encloses_all_roots_and_derivatives|Does the derivative enclosure cover a whole curved region?|Compare interval bounds with independent circle derivatives across both branches; existence remains a separate caller obligation.
implicit_curve_refuses_zero_denominator_and_cancellation|Are implicit-function prerequisites enforced?|Reject a derivative interval containing zero and propagate cancellation.
weighted_moment_encloses_independent_piecewise_polynomial|Are signed weighted kernel moments correct?|Compare with an independently expanded piecewise polynomial integral for five kernel orders and four weights.
weighted_taylor_rejects_unproved_center_and_negative_remainder|Can unsupported Taylor evidence enter the proof?|Reject a center outside the guaranteed common support and a negative remainder bound.
certified_outside_crossing_is_exactly_clipped|Can a crossing proven outside retain spurious uncertainty?|Check both monotone directions and both endpoints against exact clipped roots.
bounded_root_encloses_independent_quadratic_formula|Does bounded refinement retain the actual root?|Enclose the closed quadratic root in both orientations with three certified inclusions.
bounded_root_refuses_unknown_sign_and_preserves_cancellation|Does refinement enforce its assumptions?|Reject unknown monotonicity and propagate cancellation.
fixed_kernel_partition_covers_every_uncertain_endpoint|Can fixed proof supports omit uncertain endpoints?|Verify exact rational coverage for two/four supports and compatible complete weighted moments.
""")

_add('packaging/test_opaque_boundary_material', 'Experimental independent-reference work only; limits and original accuracy acceptance remain unchanged.', """
transport_keeps_independent_log_squared_rectangle|Does boundary Taylor transport preserve the integral?|Compare one/two/four fixed proof supports, both tensor axes and four orientations with an independent closed log-squared integral.
transport_rejects_disjoint_integrated_evidence|Can incompatible integrated evidence be accepted?|Require refusal when an integrated interval contradicts the conservative jump bound.
transport_preserves_discontinuous_and_refusal_controls|Are all jumps and refusals retained?|Preserve sixteen steps, simultaneous jumps, unknown-sign refusal and cancellation.
curved_boundary_coefficient_and_other_role_residual_match_closed_cubic|Is another angular discontinuity falsely treated as smooth?|Verify a curved coefficient against an independent cubic formula and separately retain the exact nonzero other-role residual.
conditioned_root_refinement_encloses_exact_quadratic_root|Does fixed-line refinement retain the true crossing?|Compare three inclusions with an independent quadratic root and reject a larger iteration count.
""")

_add('packaging/test_opaque_boundary_coefficient_diagnostic', 'Experimental independent-reference work only; limits and original accuracy acceptance remain unchanged.', """
saved_ninth_boundary_coefficient|Does the conditional boundary proof help the difficult region?|Measure only the recorded variant on saved regionB, retaining complete directional widths and per-knot evidence under20seconds/2048inputs.
""")

_add('packaging/test_opaque_arb_jets', 'Experimental independent-reference work only; limits and original accuracy acceptance remain unchanged.', """
compiled_primal_and_tangent_coefficients_match_independent_formulas|Are compiled series coefficients correct?|Compare reciprocal, square-root and log primal/tangent coefficients through order10 with independent90digit formulas.
compiled_convolution_contains_exact_rational_dual_polynomial|Does compiled convolution preserve every coefficient?|Compare complete spatial and first-tangent products with independent exact rational convolution.
compiled_interval_and_tiny_tangent_coefficients_are_not_dropped|Are intervals and tiny nonzero tangents preserved?|Check independently known reciprocal coefficients throughout the input interval, including a1e-70tangent.
compiled_context_restores_methods_settings_and_cancellation|Can an experiment leak into later work?|Verify method and arithmetic-context restoration after cancellation, enforce one native thread and reject nesting.
compiled_invalid_domains_refuse|Does the backend reject invalid analytic domains?|Refuse reciprocal-zero, nonpositive logarithm and invalid square-root domains.
endpoint_convolution_keeps_tight_signed_interval_bounds|Does the endpoint variant preserve tight signed bounds?|Compare all cubic coefficients with exact independently calculated interval extrema; retain the historical fixture-arithmetic failure.
""")

_add('packaging/test_opaque_arb_diagnostic', 'Experimental independent-reference work only; limits and original accuracy acceptance remain unchanged.', """
saved_ninth_compiled_boundary_coefficient|Does compiled arithmetic produce a useful full bound?|Distinguish a completed diagnostic from a successful enclosure: the first ball profile refused monotonicity; the endpoint profile reproduced the width but was slower. No whole-piece certificate.
""")

# The compiled-profile module reuses these independent controls, not their results.
for _module in ['boundary_taylor','boundary_material','high_peano','high_material']:
    _prefix='tests/packaging/test_opaque_'+_module+'.py::'
    for _key,_guide in list(GUIDES.items()):
        if _key.startswith(_prefix):
            GUIDES['tests/packaging/test_opaque_arb_material.py::'+_key[len(_prefix):]]={**_guide,'limits':_guide['limits']+' Run under the recorded isolated arithmetic profile; restore methods/settings afterwards.'}

_add('packaging/test_opaque_gauss8', 'Experimental independent-reference qualification only; all original accuracy and resource limits remain unchanged.', """
roots_are_unique_positive_weights_and_all_exact_moments|Does the eight-point rule integrate the promised polynomials?|Enclose all Legendre roots from exact rational sign brackets, require positive weights and moments through degree15, and reject degree16 exactness.
construction_is_cancellable_and_returns_owned_intervals|Can preparation overrun cancellation or leak mutable cached state?|Cancel root construction and independently rebuild after mutating earlier returned intervals.
explicit_context_restores_and_does_not_patch_old_producer|Can an experimental rule affect ordinary rendering or later tests?|Reject nested contexts, restore every selected helper after an exception and leave the old producer rule untouched.
complete_identity_keeps_exact_truncated_polynomial|Are shading boundaries still included with more samples?|Compare the complete piecewise Peano identity with independent truncated-polynomial integrals for orders2,3,6,10 and degrees0through12.
fixed_work_charge_and_attempted_work_never_exceed_256|Is extra sampling charged honestly?|Charge three units for64instead of25samples;85attempts cost255 and the next is refused under256.
cannot_run_any_region_without_full_charge|Can an underfunded region start?|Reject work before evaluation when fewer than three units remain.
""")

_add('packaging/test_opaque_gauss8_material', 'Experimental independent-reference qualification only; all original accuracy and resource limits remain unchanged.', """
lower_orders_retain_independent_material_integral|Do cheaper derivative proofs still contain the correct answer?|Compare orders3,4,5 and both directions with the independent closed logarithmic-material integral; correctness does not imply adequate width.
""")

_add('packaging/test_opaque_gauss8_square', 'Experimental independent-reference qualification only; all original accuracy and resource limits remain unchanged.', """
nonunit_polynomial_values_and_tangents_are_enclosed|Does adaptive Gauss8 preserve values and gradients?|Compare a scaled bivariate polynomial and its first tangent with independent exact rational integrals.
failed_domain_attempts_consume_full_charge_before_work|Are failed numerical attempts free?|Force a domain failure and confirm full three-unit charging before further work is refused.
too_small_allowance_never_calls_field|Can evaluation happen before resource admission?|Provide only two units and require refusal before the first field callback.
rule_cannot_silently_change_or_relax_depth_and_count|Are unsupported rules or larger limits rejected?|Reject an unqualified rule,257units and depth11 before evaluation.
tiny_tolerance_cannot_admit_incomplete_cover|Can an unfinished approximation pass?|Force exhaustion with an unresolved rational integral and preserve the last complete cover without admission.
""")

_add('packaging/test_opaque_gauss8_correction', 'Experimental independent-reference qualification only; all original accuracy and resource limits remain unchanged.', """
range_first_contains_independent_material_integral|Does selective proof work preserve the integral?|Compare both tensor directions and reversed geometry with an independent closed log-squared integral.
narrow_range_skips_optional_proof_without_numerical_threshold|Can cheap evidence safely avoid extra work?|Retain an exact1e-30contribution and verify no high proof ran when the complete range already sufficed.
monotonicity_refusal_keeps_complete_range_but_disjoint_evidence_fails|Can unsupported or conflicting evidence be admitted?|Retain a conservative range on unknown monotonicity, reject contradictory bounds and propagate cancellation.
wide_region_uses_complete_aggregate_proof_instead_of_range_only|Does the repaired selector improve wide regions?|Force the wide branch, verify a complete aggregate proof was used and intersected with the full range.
""")

_add('packaging/test_opaque_gauss8_diagnostic', 'Experimental independent-reference qualification only; all original accuracy and resource limits remain unchanged.', """
saved_ninth_gauss8|Does the experimental rule help the saved difficult regions?|Measure retained smooth and complete residual bounds under20seconds/2048inputs. A diagnostic pass records execution, not piece admission.
""")

_add('packaging/test_opaque_gauss8_preflight', 'Experimental independent-reference qualification only; all original accuracy and resource limits remain unchanged.', """
original_ninth_piece_gauss8_preflight|Can the ninth original piece pass all unchanged accuracy limits?|Verify frozen source/input hashes, execute once with256chargedunits/depth10/90reference seconds, retain terminal failures and require a complete accurate enclosure. This is an isolated preflight, not all13pieces.
""")

for _key,_guide in list(GUIDES.items()):
    _prefix='tests/packaging/test_opaque_boundary_material.py::'
    if _key.startswith(_prefix):GUIDES['tests/packaging/test_opaque_gauss8_material.py::'+_key[len(_prefix):]]={**_guide,'limits':_guide['limits']+' Explicit experimental Gauss8 context; restore after each test.'}

_add('packaging/test_opaque_angular_primitives', 'Experimental fixed-geometry checker evidence only; all original accuracy, worker and history limits remain unchanged.', """
primitives_keep_every_step_horizon_and_orientation|Are all angular branches included?|Compare orders1through6 with exact rational integrals, including horizon, discontinuities, reversed limits and zero length.
log_cubic_primitives_match_independent_exponential_moments|Are logarithmic primitives correct?|Compare each order with independent90digit exponential-moment formulas across multiple table branches.
uncertain_arguments_enclose_all_points_across_knots|Do uncertain endpoints retain branch uncertainty?|Enclose a range of anchors and endpoints crossing knots, then compare the enclosed point integrals.
primitives_refuse_bad_domains_and_preserve_cancellation_and_snapshot|Can invalid or mutated evidence enter the calculation?|Reject invalid orders/domains, propagate cancellation and preserve the original coefficient snapshot.
residual_primitives_retain_exact_same_branch_zero_and_nonzero_changes|Can a small material change disappear?|Keep exact same-branch zero while retaining every nonzero branch difference against a rational integral.
small_residual_amplitude_survives_uncertain_endpoints|Does subtraction lose tiny residuals?|Retain a1e-30contribution with outward amplitude bounds rather than dropping it by threshold.
centered_primitive_bound_preserves_cancellation_over_many_knots|Can a narrow endpoint interval stay precise?|Compare a centered integral remainder with exact tiny-residual integrals while retaining the complete local range.
residual_family_preserves_all_endpoints_and_reuses_finite_moments|Does eliminating repeated work change the proof?|Require identical interval endpoints with at most three moment vectors per call; no growing global cache or mutable cross-call alias.
""")

_add('packaging/test_opaque_primitive_parts', 'Experimental fixed-geometry checker evidence only; all original accuracy, worker and history limits remain unchanged.', """
all_boundary_terms_preserve_independent_step_polynomial_integral|Are every boundary term and orientation correct?|Compare orders1through6 and both directions against exact rational discontinuous-material integrals.
nonlinear_coordinate_keeps_jacobian_and_complete_remainder|Does a curved coordinate retain its Jacobian?|Compare a nonlinear change of variable with an independent exact transformed integral.
unknown_coordinate_direction_and_parameter_tangent_are_refused|Can unsupported geometry be certified?|Refuse uncertain monotonicity, parameter tangents and missing derivative orders.
parts_propagates_cancellation|Can analytic work ignore a stop?|Require cancellation to escape the integration-by-parts calculation.
missing_highest_coordinate_derivative_is_rejected|Can truncating the coordinate hide a remainder?|Require the extra coordinate derivative needed for every retained integration-by-parts order.
nonzero_remainder_is_required_for_exact_rational_integral|Can boundary terms alone falsely pass?|Use known rational integrals for which the boundary term is wrong and the complete remainder is essential.
""")

_add('packaging/test_opaque_primitive_rectangle', 'Experimental fixed-geometry checker evidence only; all original accuracy, worker and history limits remain unchanged.', """
primitive_rectangle_contains_closed_one_coordinate_material|Does the rectangle identity contain the actual integral?|Compare both axes and orientations, orders3and6, with an independent closed logarithmic-material integral.
product_residual_is_separate_and_never_discarded|Are simultaneous material changes preserved?|Retain the product residual separately and use a complete range if directional monotonicity is unknown.
line_requires_fixed_other_coordinate_and_keeps_independent_integral|Are line and area integrals distinguished?|Require an exact transverse point for a line and check its independent integral; reject a zero-area rectangle.
""")

_add('packaging/test_opaque_primitive_hybrid', 'Experimental fixed-geometry checker evidence only; all original accuracy, worker and history limits remain unchanged.', """
hybrid_keeps_exact_swept_log_cubic_integral|Does analytic line integration justify a complete region?|Combine enclosed lines with the full outer Peano error, including all jumps; compare both axes and orientations with an independent closed integral.
trace_encloses_every_transverse_point_and_keeps_its_dimension|Can uncertain Gauss nodes be treated as exact?|Retain node intervals in line traces while preserving the one-dimensional integration measure.
hybrid_propagates_cancellation_and_refuses_invalid_axis|Can the hybrid bypass its supported domain or stop?|Reject unsupported axes and propagate cancellation before analytic work continues.
""")

_add('packaging/test_opaque_primitive_diagnostic', 'Experimental fixed-geometry checker evidence only; all original accuracy, worker and history limits remain unchanged.', """
saved_ninth_primitive_identity|Does the analytic method help the troublesome original region?|Measure full bounds and both directions with all preparation inside20seconds/2048inputs. Execution success is not numerical admission; wide bounds and failed diagnostics remain visible.
""")

_add('packaging/test_opaque_material_inputs', 'Experimental checker evidence only; original accuracy, resource caps and renderer are unchanged.', """
material_only_retains_exact_cosine_endpoints_and_derivatives|Does skipping unused specular work change material inputs?|Require exactly matching cosine coefficients throughorder10, including tangents and both horizon sides.
half_vector_sqrt_work_is_absent_only_when_explicitly_selected|Is unnecessary work actually removed?|Count four square roots for full inputs and two for explicit material-only inputs; absent half-vector is None, never a fake zero.
material_only_preserves_horizon_mismatch_and_domain_refusals|Can an optimized input bypass the horizon contract?|Retain the certified-patch and horizon-distance mismatch refusal.
original_material_profile_separates_specular_calls_and_cache_keys|Can material caching corrupt original specular rendering?|Use original mesh data, both qualified coordinate maps and several derivative orders; full calls retain the half-vector and unchanged cosine/Jacobian endpoints.
material_cache_preserves_explicit_missing_half_vector_and_owned_jets|Does reuse preserve absent and mutable values?|Keep the explicit missing half-vector while returning independent cached jets; retain the original discovered cache failure.
original_profile_refuses_unqualified_coordinate_powers|Can a different map enter silently?|Refuse unsupported powers and noninteger selectors before evaluating the original scene.
""")

_add('packaging/test_opaque_material_input_cost', 'Experimental checker evidence only; original accuracy, resource caps and renderer are unchanged.', """
saved_regions_keep_exact_bounds_without_unused_half_vector|Does the input optimization improve the actual checker?|Compare three saved regions in alternating order with exact full bounds, separate input time and all preparation charged. First-use costs are retained and not advertised as speedup.
""")

_add('packaging/test_opaque_affine_feasibility', 'Experimental checker evidence only; original accuracy, resource caps and renderer are unchanged.', """
saved_regions_affine_control_error_budget|Would a simple analytic approximation be accurate enough?|Bound geometry and weight remainders on three saved regions. This is a feasibility budget, not an integral certificate; large bounds and domain refusals prevent adoption.
""")

_add('packaging/test_opaque_sharp_range', 'Experimental checker evidence only; original accuracy, resource caps and renderer are unchanged.', """
individual_segments_retain_independent_high_derivatives|Do individual angular segments preserve all derivatives?|Compare throughorder10 with independent90digit log-polynomial derivative formulas; intersect with the original complete range.
exact_same_branch_zero_and_horizon_are_retained|Does refinement retain exact zero and the horizon?|Keep genuine same-branch zeros throughorder10, including the linear horizon.
sharp_ranges_keep_snapshot_cancel_and_domain_refusal|Can refined ranges lose provenance or ignore cancellation?|Preserve the immutable resource snapshot and invalid-domain/stop behavior.
fixed_geometry_sharp_bounds_refuse_parameter_tangents|Can an unsupported moving-geometry derivative pass?|Refuse tangents even at orderzero; preserve the discovered failure and repaired check.
sharp_full_material_keeps_independent_log_integral|Does the refined index work in a complete material integral?|Compare all orientation combinations with an independent closed logarithmic integral.
sharp_full_material_keeps_simultaneous_product_jumps|Can simultaneous shading changes lose a cross term?|Retain the complete independent product-jump control with the refined index.
""")

_add('packaging/test_opaque_sharp_diagnostic', 'Experimental checker evidence only; original accuracy, resource caps and renderer are unchanged.', """
saved_regions_compare_grouped_and_individual_segment_bounds|Are tighter segment bounds worth their cost?|Measure all three saved regions under20seconds/2048inputs; narrower bounds alone do not justify a whole-piece trial.
""")

_add('packaging/test_opaque_map_aggregate_diagnostic', 'Experimental checker evidence only; original accuracy, resource caps and renderer are unchanged.', """
eighth_power_map_with_complete_aggregate_bound|Does combining the older coordinate map with the newer proof help?|Use outward covers of saved physical regions with the full Jacobian and all smooth/residual errors. Results do not improve enough to select this map.
""")

_add('packaging/test_opaque_binary_derivatives', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
exact_fraction_interval_convolution|Do fast bounds contain exact arithmetic?|Compare independent rational convolutions and unchanged low coefficients.
extremes_preserve_tiny_terms_or_fallback|Can extreme values disappear?|Keep tiny terms and require fallback for unsupported ranges.
zero_sparse_scalars_low_orders_and_mismatch|Are simple paths preserved?|Check zeros, scalar operations and order mismatch refusal.
context_restores_after_exception_and_rejects_nesting|Can an experiment leak into later work?|Require method restoration and reject nested contexts.
""")

_add('packaging/test_opaque_binary_material', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
binary_high_material_closed_integral|Do complete integrals remain enclosed?|Check independent closed integrals across signs, axes and orders.
binary_material_jumps_and_refusals|Are shading discontinuities retained?|Check simultaneous jumps and domain refusals.
""")

_add('packaging/test_opaque_binary_diagnostic', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
saved_regions_binary_high_derivative_cost|Does the arithmetic change help the checker?|Alternate profiles on three frozen regions under20seconds/2048inputs; no consistent useful gain was found.
""")

_add('packaging/test_opaque_kernel_power_reuse', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
reused_powers_exact_endpoints|Does local power reuse change a bound?|Compare all endpoints to the old formula throughdegree10.
polynomial_power_work_is_linear|Was redundant work removed?|Count power evaluation without enlarging persistent storage.
""")

_add('packaging/test_opaque_kernel_boundary_profile', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
saved_boundary_with_reused_kernel_powers|Where does the sharper checker spend time?|Profile one region within20seconds/2048inputs; instrumented timings are not normal execution speed.
""")

_add('packaging/test_opaque_arb_intervals', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
persistent_intervals_contain_exact_endpoint_arithmetic|Are compiled endpoints outward?|Check exact rational endpoint extrema, mixed signs, tiny products and invalid division.
persistent_analytic_jets|Are values and derivatives enclosed?|Compare independent90digit formulas throughorder10 and their parameter tangents.
context_tiny_zero_domains_and_restoration|Can unsupported arithmetic pass?|Require tiny-term preservation, genuine zero, domain refusal and restoration.
exact_authored_decimal_center_keeps_its_known_identity|Can conversion change an exact input centre?|Preserve its known exact bounds while computed arithmetic remains outward; retain the initial failure.
copy_does_not_force_decimal_export|Can persistent copies avoid conversion work?|Keep owned dyadic endpoints without forced Decimal exports.
persistent_series_contains_exact_interval_convolution|Does native series arithmetic preserve signed bounds?|Check every coefficient against independent exact rational convolution.
""")

_add('packaging/test_opaque_persistent_material', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
persistent_high_material_closed_integral|Does the backend preserve a complete material integral?|Check closed integrals across both directions and derivative orders.
persistent_material_jumps_and_refusals|Does the backend retain shading boundaries?|Check all jumps and unsupported-domain refusals.
""")

_add('packaging/test_opaque_persistent_diagnostic', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
saved_ninth_persistent_endpoints|Is the compiled backend useful on the original mesh?|Retain the refused centre experiment and repaired bounds; no useful speed gain or mesh admission was established.
""")

_add('packaging/test_opaque_radial_reuse', 'Bounded checker experiment; original accuracy remains unadmitted, all limits and failures retained.', """
shared_radial_root_preserves_all_coefficients|Can geometry reuse one conic root?|Require exact coefficients throughorder10, both axes, tangents, horizon sides and maps.
shared_root_computes_discriminant_once|Is duplicate root work removed?|Require one discriminant square root and unchanged Jacobian and numerator.
""")

_add('packaging/test_opaque_kernel_cost', 'Historical profiling receipt from opaque109; the older source file was restored and the new test moved to test_opaque_kernel_boundary_profile.py.', """
saved_boundary_with_reused_kernel_powers|What does this historical profile measure?|One saved boundary region; original test hash retained, no report relabelled and no whole-piece admission.
""")

_add('packaging/test_opaque_signed_prefix_diagnostic', 'Bounded proof experiment; no original accuracy admission or changed resource limits.', """
saved_regions_signed_jump_prefix_feasibility|Can signed table jumps cancel before integration?|Inspect three saved regions under20seconds/2048inputs. All view jumps have the same sign; raw prefix cancellation is not useful.
""")

_add('packaging/test_opaque_boundary_order4', 'Bounded proof experiment; no original accuracy admission or changed resource limits.', """
fourth_circle_coefficient_independent_formula|Is the fourth implicit derivative correct?|Check both circle branches against an independent closed formula.
order4_complete_material_closed_integral|Does the higher Taylor order preserve full integration?|Compare closed log-material integrals in both axes and orientations.
order4_is_explicit_and_invalid_degrees_refuse|Can an unqualified order enter silently?|Retain defaultorder3 and reject unsupported orders.
order3_identity_retains_complete_material|Does a lower complete identity remain valid?|Check closed integrals before measuring its larger original-region uncertainty.
""")

_add('packaging/test_opaque_boundary_order4_diagnostic', 'Bounded proof experiment; no original accuracy admission or changed resource limits.', """
saved_ninth_fourth_order_boundary|Does one more boundary Taylor term solve the bottleneck?|Measure the complete region with all terms; only a small bound reduction occurs, insufficient for adoption.
""")

_add('packaging/test_opaque_boundary_local_base', 'Bounded proof experiment; no original accuracy admission or changed resource limits.', """
local_extension_keeps_closed_integral|Can the local smooth branch change the integral?|Require independent closed integrals while retaining the entire other-role residual.
local_base_is_explicit_and_domain_checked|Can local branch selection become an implicit default?|Keep it opt-in and refuse invalid selectors.
explicit_local_base_preserves_complete_value_and_residual|Can changing the smooth extension discard a contribution?|Verify smooth plus residual against the independent exact value, including genuine zero residual.
""")

_add('packaging/test_opaque_boundary_local_diagnostic', 'Bounded proof experiment; no original accuracy admission or changed resource limits.', """
saved_ninth_local_extension_boundary|Does a local branch tighten the troublesome bound?|Charge1980inputs and all setup; the negligible gain does not qualify the original piece.
""")

_add('packaging/test_opaque_boundary_terms_diagnostic', 'Bounded proof experiment; no original accuracy admission or changed resource limits.', """
saved_boundary_remainder_terms|Which retained error terms actually dominate?|Observe existing coefficient and weighted-kernel calculations without extra integrand calls; distinguish curvature, higher jumps and ordinary error.
""")

_add('packaging/test_opaque_boundary_order3_diagnostic', 'Bounded proof experiment; no original accuracy admission or changed resource limits.', """
saved_ninth_order3_complete_boundary_identity|Can a lower-order complete proof avoid loose higher-jump terms?|Retain the ordinary remainder and every required jump; the observed bound is much wider, so this route is not selected.
""")

_add('packaging/test_opaque_boundary_jump_localization', 'Bounded checker experiment; no whole-pixel admission or changed acceptance.', """
localized_higher_jumps_keep_independent_closed_integral|Does localizing the remaining jumps preserve the integral?|Check closed logarithmic integrals in both axes, orientations and support partitions.
localized_higher_jumps_keep_discontinuities_and_refusals|Are all discontinuities still included?|Retain steps, simultaneous branches and unsupported-domain refusals.
higher_localization_is_explicit|Can this experiment silently become the default?|Require explicit selection and keep the existing default.
curved_higher_jump_integral_matches_independent_polynomial|Are curved higher jumps actually bounded?|Integrate independently derived polynomial coefficients with exact kernel moments in both axes.
localized_jump_failure_boundaries|Can failed geometry produce a false pass?|Require exact fallback only for missing monotonicity, disjoint-evidence refusal and propagated cancellation.
""")

_add('packaging/test_opaque_boundary_jump_diagnostic', 'Bounded checker experiment; no whole-pixel admission or changed acceptance.', """
saved_ninth_localized_higher_jumps|Does the complete stronger proof help the saved region?|Retain every higher jump; compare the narrower interval and charged cost without claiming whole-piece admission.
""")

_add('packaging/test_opaque_boundary_jump_regions', 'Bounded checker experiment; no whole-pixel admission or changed acceptance.', """
saved_companion_region_localized_higher_jumps|Does the repair help the other saved regions?|Evaluate fresh bounds under20seconds/2048inputs. Keep the older bound when stronger; distinguish smooth and correction uncertainty.
""")

_add('packaging/test_opaque_seeded_cover', 'Bounded checker experiment; no whole-pixel admission or changed acceptance.', """
seeded_cover_is_complete_disjoint_and_fresh|Does the seed grid cover exactly the full domain?|Check disjoint rational rectangles, total area, fresh evaluations, exact integrals and tangents.
seeded_cover_refuses_invalid_shapes|Can arbitrary malformed seeds enter?|Reject invalid dimensions and require exact power-of-two grid counts.
seeded_cover_checks_work_and_depth_before_field|Can seeding evade resource limits?|Charge every fresh region and inherited binary depth before any field evaluation.
seeded_cover_has_independent_nonlinear_value_and_tangent|Does seeded integration enclose a nonlinear result?|Compare independent90digit closed formulas for value and tangent.
seeded_cover_never_accepts_partial_cover_or_ignores_cancellation|Can an unfinished cover be accepted?|Interrupt during seeding and require no complete enclosure or false pass.
seeded_cover_budget_failure_retains_only_complete_fresh_bounds|Does exhaustion preserve honest evidence?|Retain only the complete fresh cover and refuse its excessive uncertainty.
""")

_add('packaging/test_opaque_seeded_preflight', 'Bounded checker experiment; no whole-pixel admission or changed acceptance.', """
original_ninth_piece_fresh_seeded_preflight|Can a fresh grid finish the ninth piece within its limits?|Pin all source and input bytes; evaluate every region anew under the unchanged256cells/depth10/90seconds. Preserve failed attempts; this alone cannot admit the complete pixel.
""")

_add('packaging/test_opaque_direct_aggregate', 'Checker experiment; original accuracy remains unadmitted and all limits remain.', """
direct_bound_encloses_independent_closed_integral|Does the direct correction still enclose the full material?|Compare independent closed integrals across both signs with the range and aggregate proofs.
direct_bound_matches_existing_fallback_exactly_without_preliminary_work|Was only unnecessary preliminary work removed?|Require identical endpoints and fewer preliminary calls.
direct_bound_retains_stop_and_domain_boundaries|Can unsupported proof data pass?|Retain monotonicity fallback, disjoint-evidence refusal and cancellation.
""")

_add('packaging/test_opaque_direct_aggregate_cost', 'Checker experiment; original accuracy remains unadmitted and all limits remain.', """
saved_regions_direct_complete_aggregate_cost|Does direct evaluation save work on the original regions?|Use the same three boxes and exact bounds; separate cold setup from warm timing.
""")

_add('packaging/test_opaque_direct_preflight', 'Checker experiment; original accuracy remains unadmitted and all limits remain.', """
original_ninth_piece_direct_aggregate_preflight|Can cheaper correction finish the ninth piece?|The frozen trial reaches256evaluations in about41seconds, but its accuracy bound is still too wide. No admission.
""")

_add('packaging/test_opaque_persistent_series', 'Checker experiment; original accuracy remains unadmitted and all limits remain.', """
persistent_series_convolution_encloses_exact_rational_endpoints|Are compiled derivative products outward?|Compare every coefficient and tangent with independent rational endpoint products.
persistent_series_analytic_coefficients_and_tangent|Are analytic derivatives and tangents enclosed?|Compare independent closed formulas throughorder10.
persistent_series_mutable_coefficients_tiny_terms_and_order_checks|Can mutation or tiny terms corrupt a series?|Verify invalidation, nonzero tiny tangents and order refusal.
persistent_series_domains_context_restoration_and_nesting|Can the experimental backend leak or ignore domains?|Reject invalid domains and nesting, and restore state after exceptions.
persistent_series_interval_analytic_derivatives|Do derivative bounds cover an interval of inputs?|Check independent endpoint formulas for reciprocal, square root and logarithm.
persistent_series_exact_zero_and_signed_sparse_algebra|Are genuine zeros and signed operations preserved?|Check exact cancellation, zero products and scalar algebra.
persistent_series_retains_independent_constant_range_identity|Can broad balls break a valid input range?|Retain independently directed value bounds instead of treating rounded overshoot as physical data.
persistent_series_extreme_exponent_conversion_is_bounded_and_nonzero|Can tiny conversion values consume the worker budget?|Use bounded decimal and dyadic conversion without dropping nonzero terms; preserve the earlier timeout.
persistent_series_preserves_original_low_coefficients_exactly|Are the low derivatives unchanged?|Require exact original value and tangent endpoints through spatial order2.
""")

_add('packaging/test_opaque_series_material', 'Checker experiment; original accuracy remains unadmitted and all limits remain.', """
series_localized_material_keeps_closed_integral|Does compiled high arithmetic preserve the whole integral?|Check independent logarithmic material integrals in both directions and orientations.
series_curved_jump_integral_matches_independent_polynomial|Does it preserve curved boundary terms?|Compare an independently derived polynomial integral for every retained higher jump.
series_localized_material_keeps_discontinuities|Are steps and simultaneous transitions retained?|Keep all contributions and refusals. Initial range errors and the conversion deadline remain failed evidence.
""")

_add('packaging/test_opaque_series_diagnostic', 'Checker experiment; original accuracy remains unadmitted and all limits remain.', """
saved_boundary_persistent_series_cost_and_bound|Does compiled high arithmetic help the mesh checker?|The first profile refuses monotonicity; preserving original low coefficients restores a bound, but it is slower and wider. This backend is rejected for this mesh.
""")

_add('packaging/test_opaque_algebraic_boundary', 'Experimental checker controls; original accuracy and M1 remain unfinished.', """
algebraic_implicit_curve_matches_independent_reciprocal|Does the algebraic boundary follow the same curve?|Compare each implicit derivative with an independent reciprocal formula on the positive branch.
role_only_cosine_retains_every_original_coefficient|Can one cosine be computed without changing its evidence?|Require exact original endpoints for values and derivatives.
algebraic_curve_range_encloses_independent_derivatives|Does the bound cover the entire input interval?|Enclose closed-form derivatives at both endpoints, not just one sample.
algebraic_boundary_rejects_wrong_branch_and_invalid_role|Can squaring admit the wrong root?|Reject negative numerators, invalid roles and nonpositive knots.
horizon_role_only_retains_exact_coefficients|Does the horizon mapping preserve the original cosine?|Check both horizon sides and all tested derivative orders.
""")

_add('packaging/test_opaque_algebraic_material', 'Experimental checker controls; original accuracy and M1 remain unfinished.', """
algebraic_boundary_keeps_independent_material_integral|Does the whole material integral remain correct?|Check independent logarithmic integrals in both directions and signs, retaining all jumps.
algebraic_boundary_fallback_is_specific_and_cancellation_propagates|Can a failed algebraic proof silently pass?|Allow only declared geometric fallback; propagate cancellation and corrupt equations.
algebraic_boundary_is_opt_in_and_validated|Can the experiment change default acceptance?|Require explicit Boolean selection; keep the default producer unchanged.
boundary_does_not_require_unused_specular_half_vector|Can the diffuse proof omit unused specular work?|Six closed-integral controls retain all required cosine and Jacobian terms. The earlier missing-value failure is preserved.
""")

_add('packaging/test_opaque_algebraic_binding', 'Experimental checker controls; original accuracy and M1 remain unfinished.', """
original_algebraic_helpers_preserve_prepared_roles|Are the helpers correctly wired to the original mesh?|Check exact prepared role coefficients for full and material-only inputs.
""")

_add('packaging/test_opaque_algebraic_diagnostic', 'Experimental checker controls; original accuracy and M1 remain unfinished.', """
saved_ninth_algebraic_boundary|Does the cheaper equation help the difficult mesh region?|Charge all full, role and implicit evaluations. The final boundary calculation is4.81seconds versus earlier5.44seconds, but its bound is wider. This is local evidence, not a piece certificate.
""")

_add('packaging/test_opaque_curve_prefix', 'Experimental checker controls; original accuracy and M1 remain unfinished.', """
implicit_equation_never_computes_unused_higher_coefficients|Can unnecessary derivative work be removed?|Compute only each required prefix and verify independent reciprocal coefficients.
""")

_add('packaging/test_opaque_native_convolution', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
exact_fraction_interval_convolution|Are native interval products outward?|Compare every coefficient and tangent against exact rational endpoint products.
extremes_preserve_tiny_terms_or_fallback|Can extreme values disappear?|Retain nonzero terms or use the original arithmetic on unsupported ranges.
zero_sparse_scalars_low_orders_and_mismatch|Are zero, sparse and low-order products preserved?|Check exact zero, scalar multiplication and mismatched-order refusal.
context_restores_after_exception_and_rejects_nesting|Can an arithmetic experiment leak into later work?|Restore the original methods after exceptions and refuse nesting.
native_abi_refuses_invalid_data|Does the compiled boundary reject malformed input?|Reject invalid sizes, coordinates, duplicate entries, NaNs and reversed intervals.
native_manifest_hash_refusal|Is the native artifact pinned?|Refuse a mismatched manifest digest.
native_short_series_contains_exact_convolution|Are short derivative series also outward?|Check exact rational products and unchanged low coefficients at orders3through5.
""")

_add('packaging/test_opaque_native_material', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
algebraic_boundary_keeps_independent_material_integral|Does the compiled checker keep complete material integrals?|Compare independent closed integrals with both axes and signs, retaining all boundary terms.
algebraic_boundary_fallback_is_specific_and_cancellation_propagates|Can a failed proof pass through fallback?|Only declared geometry refusals may fall back; cancellation and corrupt equations propagate.
algebraic_boundary_is_opt_in_and_validated|Can an experiment silently change the default?|Require explicit selection; default production acceptance remains unchanged.
boundary_does_not_require_unused_specular_half_vector|Does omitting unused specular work preserve the proof?|Verify closed integrals with the optional half-vector absent.
""")

_add('packaging/test_opaque_native_diagnostic', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
saved_ninth_native_convolution|Did compiled binary convolution help the mesh proof?|It passed controls but was slower than Decimal on the saved region and was rejected.
""")

_add('packaging/test_opaque_decimal_native', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
compiled_decimal_preserves_endpoint_and_iteration_identity|Does compiled bookkeeping retain the same arithmetic?|Require identical endpoints and coefficient iteration order for shuffled signed inputs at every selected order.
compiled_decimal_extremes_and_exact_zero|Are tiny and large Decimal terms retained?|Compare exact rational results and original endpoints; never drop a nonzero term.
compiled_decimal_context_restoration_and_mismatch|Does the original arithmetic return after an exception?|Check restoration, nesting refusal and order mismatch.
decimal_native_rejects_invalid_coefficients|Can malformed coefficient lists enter the loop?|Reject invalid counts, axes, tuples, duplicate keys and orders.
decimal_native_preserves_exception_and_restores_context|Are Decimal traps and rounding context preserved?|Force an overflow and verify original exception and context restoration.
decimal_native_rejects_manifest_tampering|Is the compiled Decimal module pinned?|Reject a changed manifest hash before loading.
decimal_native_keeps_scalar_overhead_on_original_path|Can scalar overhead be avoided without changing results?|Use the original loop for a single product and require exact endpoint identity.
""")

_add('packaging/test_opaque_decimal_material', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
algebraic_boundary_keeps_independent_material_integral|Does the compiled checker keep complete material integrals?|Compare independent closed integrals with both axes and signs, retaining all boundary terms.
algebraic_boundary_fallback_is_specific_and_cancellation_propagates|Can a failed proof pass through fallback?|Only declared geometry refusals may fall back; cancellation and corrupt equations propagate.
algebraic_boundary_is_opt_in_and_validated|Can an experiment silently change the default?|Require explicit selection; default production acceptance remains unchanged.
boundary_does_not_require_unused_specular_half_vector|Does omitting unused specular work preserve the proof?|Verify closed integrals with the optional half-vector absent.
""")

_add('packaging/test_opaque_decimal_diagnostic', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
saved_ninth_decimal_convolution|Does exact compiled bookkeeping help the saved mesh region?|The selected hybrid takes4.42boundary seconds with identical endpoints; this is local evidence, not whole-piece qualification.
""")

_add('packaging/test_opaque_decimal_cost', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
decimal_convolution_crossover_is_bounded_and_exact|When is the compiled loop worth calling?|A bounded960product matrix retains exact endpoints and measures both execution orders.
""")

_add('packaging/test_opaque_selective_boundary', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
selective_boundary_preserves_both_proofs_and_stopping|Can selection substitute for a complete bound?|Intersect only independently complete proofs; retain fallback, disjoint refusal and cancellation.
selective_boundary_encloses_independent_complete_log_integral|Does the selective strategy preserve the complete integral?|Use closed logarithmic integrals across signs and axes with both original and compiled arithmetic.
""")

_add('packaging/test_opaque_selective_preflight', 'Checker qualification only; original accuracy and M1 remain unfinished.', """
original_ninth_piece_selective_boundary_preflight|Can the stronger selective proof finish the ninth piece?|Use fresh frozen sources, unchanged90second256cell limits and preserved failure evidence. A preflight cannot be reused as a final campaign certificate.
""")

_add('packaging/test_opaque_offline_profile', 'Offline checker qualification; full original rendering accuracy and M1 remain open.', """
profile_is_explicit_immutable_and_exact|Can offline validation change rendering limits?|Keep named immutable checker profiles; verify both exact limit sets and unchanged one-thread/one-GiB limits.
unknown_profile_fails_closed|Can an unknown profile silently receive more time?|Refuse missing, malformed and unknown profile identities.
larger_profile_still_refuses_growth_before_field|Does the approved profile still have firm limits?|Reject excess cells, depth and malformed counts before evaluating the field.
larger_seed_cover_is_exact_and_charged|Does more subdivision preserve the integral?|Cover the unit square with512fresh disjoint cells and enclose the exact polynomial integral; count every cell.
extended_depth_charges_failed_domains_and_stops|Can domain failures evade the depth limit?|Charge failed regions and stop at the approved depth, without returning an approximate answer.
extended_cell_cap_charges_failed_domains_and_stops|Can failed evaluations evade the work cap?|Charge attempted cells and retain an explicit unresolved result at the cap.
deadline_checks_after_last_field_not_just_between_cells|Can a late result accidentally pass?|Advance a controlled clock during the final field evaluation; both ordinary and extended profiles must reject it.
profile_does_not_relax_closed_integral_tangent_or_accuracy|Does a larger resource budget weaken accuracy?|Enclose independent exact value and tangent integrals under both profiles; reject a loose requested tolerance.
default_caps_and_environment_do_not_select_extended|Can environment variables raise ordinary limits?|An unrelated environment setting must not select the extended checker.
worker_profile_cannot_raise_rendering_or_other_test_limits|Can an offline exception leak into rendering tests?|Refuse the extended profile on rendering, CUDA, ordinary campaign and teaching modes.
worker_limits_bind_the_single_offline_mode|Is the extended worker explicitly selected?|Require the offline diagnostic mode and its matching profile.
supervisor_refuses_resource_and_identity_failures|Does the supervisor stop invalid work?|Controlled process fixtures exercise watchdog, memory, exit and PID failures without waiting for real long deadlines.
worker_routes_caps_and_profile_to_supervisor|Does the launched worker receive the reviewed limits?|Inspect actual supervisor arguments and930/960second configuration.
full_piece_profile_binding_preserves_original_share|Does the ninth piece retain its accuracy allocation?|Inspect producer routing against the original conserved thirteen-piece tolerance; this mock proves routing only.
standard_worker_restores_limits_after_offline_selection|Can an offline run affect a later ordinary run?|Enter the runner twice and verify that the second invocation restores120/180second limits.
outer_checkpoint_retains_charged_work_and_complete_cover|Does cancellation preserve attempted work?|Interrupt an evaluation and require its charged-cell count in the failure evidence.
progress_reports_only_complete_cover_and_cannot_change_acceptance|Can a progress display certify a result?|Expose copied complete-cover counts and widths only; mutating the report cannot change numerical acceptance.
""")

_add('packaging/test_opaque_offline_preflight', 'Offline checker qualification; full original rendering accuracy and M1 remain open.', """
original_ninth_piece_extended_offline_preflight|Can more offline subdivisions certify the difficult ninth piece?|A fresh frozen4096cell diagnostic retains the original accuracy and input. Its first run reaches the cell cap with uncertainty still13.4253times too wide.
""")

_add('packaging/test_opaque_offline_leaf_diagnostic', 'Offline checker qualification; full original rendering accuracy and M1 remain open.', """
saved_extended_ninth_small_regions|Where should the stronger proof be used?|Compare both integration directions and smooth terms on three frozen small regions, within20seconds and2048geometry requests; diagnostic bounds are not a campaign certificate.
""")

_add('packaging/test_opaque_late_boundary', 'Offline checker qualification; full original rendering accuracy and M1 remain open.', """
late_rule_spends_only_on_small_unresolved_corrections|Can stronger proofs be delayed until they are useful?|Skip broad, horizon-adjacent or already tight corrections; use the complete stronger proof only on eligible small rectangles.
late_rule_keeps_failure_and_complete_intersection|Can selective work conceal a failed proof?|Retain cancellation, disjoint-bound refusal and explicit monotonicity fallback.
late_schedule_refuses_invalid_target|Can a malformed scheduling target weaken acceptance?|Reject invalid targets; the full summed enclosure remains the acceptance authority.
late_proof_encloses_independent_complete_log_integral|Does the late proof preserve the complete material integral?|Compare independent closed logarithmic integrals across both axes and all sign combinations with compiled Decimal arithmetic.
""")

_add('packaging/test_opaque_offline_late_preflight', 'Offline checker qualification; full original rendering accuracy and M1 remain open.', """
original_ninth_piece_late_boundary_preflight|Can late boundary integration certify the original ninth piece?|Use fresh frozen input and the separately approved offline profile. Retain every contribution, progress snapshots and terminal outcome; no automatic original-pixel admission.
""")

_add('packaging/test_opaque_extended_campaign','Separate offline certificate qualification; rendering limits stay unchanged.',"""
extended_campaign_conserves_exact_rgb_and_albedo_with_same_tolerance|Does a larger offline budget weaken the image tolerance?|Compose independent exact constant integrals and enforce the same final RGB/albedo radius; verify the approved per-piece and aggregate work limits.
extended_campaign_refuses_failed_or_overbudget_receipts|Can an over-budget or unmonitored worker be accepted?|Reject excess cells, depth, time, memory, failed exit and mismatched monitored identity.
extended_campaign_refuses_mismatch_and_counterfeit_evidence|Can partial or altered certificates pass?|Reject missing, duplicate, reordered, changed-input/source/profile, loose-bound and rehashed malformed evidence.
extended_prefix_reserves_next_whole_job_and_refuses_partial_admission|Does a partial campaign establish pixel accuracy?|Reserve a complete next job and refuse final composition before thirteen valid pieces.
standard_campaign_contract_remains_unchanged|Does the separate profile alter ordinary tests?|Keep the existing256cell/90second campaign contract and its closed-integral evidence.
extended_piece_worker_routes_only_ninth_to_late_proof|Does each piece receive the intended numerical method?|Routing-only mocks verify the ninth late proof, twelfth direction selector and thirteenth radial proof, their seeds/material-only inputs and unchanged other-piece schedule. Mock values are never numerical evidence.
extended_worker_modes_require_matching_profile|Can a worker silently select more resources?|Require a matching explicit profile for each offline worker mode.
""")

_add('packaging/test_opaque_extended_piece','Fresh original visible-piece evidence under the reviewed offline profile.',"""
extended_original_piece_independent_certificate|Does this original piece meet its conserved accuracy share?|Verify frozen source/input/native arithmetic, compute fresh complete bounds and retain all failures. Only thirteen validated receipts and their sum can qualify the selected pixel.
""")

_add('packaging/test_opaque_twelfth_leaf_diagnostic','Bounded original-piece accuracy evidence; all limits and failures retained.',"""
saved_extended_twelfth_small_regions|Which integration direction controls the saved twelfth error?|Inspect three frozen failed regions, ordinary terms and complete correction bounds in both directions, capped at20seconds and2048geometry requests. Diagnostic evidence cannot admit the whole piece.
""")

_add('packaging/test_opaque_twelfth_boundary','Bounded original-piece accuracy evidence; all limits and failures retained.',"""
twelfth_rule_spends_only_on_small_unresolved_corrections|Does radial scheduling preserve unresolved error?|Select the radial proof on small rectangles including near the horizon. A scheduling target cannot replace complete-bound acceptance.
twelfth_rule_keeps_failure_and_complete_intersection|Can failed or conflicting proofs be hidden?|Propagate cancellation, reject disjoint complete bounds and retain the original full bound when monotonicity cannot be certified.
twelfth_schedule_refuses_invalid_target|Can an invalid local target enter scheduling?|Reject zero, negative, nonfinite and excessive targets before work.
twelfth_proof_encloses_independent_complete_log_integral|Does complete boundary integration enclose a known answer?|Check independent closed logarithmic integrals in both directions and all sign orientations using the pinned compiled Decimal path.
tail_preflight_requires_explicit_offline_profile|Can a tail preflight silently inherit larger limits?|Require the exact opt-in offline profile and reject the ordinary-profile mismatch.
""")

_add('packaging/test_opaque_offline_tail_preflight','Bounded original-piece accuracy evidence; all limits and failures retained.',"""
original_tail_piece_boundary_preflight|Does a final original piece meet its unchanged accuracy share?|Verify frozen sources and input, then evaluate the twelfth radial schedule or the thirteenth original schedule under the approved offline caps. Preserve failed bounds; no diagnostic certificate substitutes for a fresh full campaign.
""")

_add('packaging/test_opaque_tail_monotonicity','Independent bounded accuracy checking; complete evidence remains required.',"""
saved_tail_monotonicity_refusals|Is derivative uncertainty a real turn or an over-wide bound?|Compare natural and midpoint/Hessian derivative inclusions on three preserved regions per tail piece. Both integration directions are examined under20seconds and2048geometry requests; local findings cannot admit the whole piece.
""")

_add('packaging/test_opaque_centered_regions','Independent bounded accuracy checking; complete evidence remains required.',"""
centered_geometry_on_saved_tail_regions|Does the centred geometric proof help the saved difficult regions?|Use the qualified geometric derivative intersection while retaining all material branches and complete correction bounds. Record both directions and refusals; a local success remains separate from whole-piece acceptance.
""")

_add('packaging/test_opaque_centered_gradient','Independent bounded accuracy checking; complete evidence remains required.',"""
centered_gradient_recovers_sign_lost_to_dependency|Can an exact derivative sign survive interval dependency?|Use a known positive or negative derivative in either coordinate and prove its centred enclosure recovers the sign.
centered_gradient_keeps_a_real_turn_uncertain|Can centering hide a true derivative turn?|A quadratic with a genuine sign change must retain zero in the derivative enclosure.
centered_gradient_bounds_nonlinear_polynomial_at_exact_points|Does the mean-value enclosure contain exact nonlinear derivatives?|Compare with independent rational polynomial derivatives across the rectangle, including both coordinates.
centered_gradient_does_not_reinterpret_composed_jets|Can unsupported derivative coordinates be relabelled?|Scaled, mixed or composed jets retain their original coefficients rather than borrowing an inapplicable partial derivative.
centered_gradient_propagates_cancellation|Can extra proof work ignore cancellation?|Every supplementary geometry evaluation checks cancellation and propagates the stop.
centered_gradient_rejects_disjoint_producer_evidence|Can incompatible geometric proofs be silently combined?|Disjoint natural and mean-value inclusions raise an error instead of choosing a convenient interval.
centered_boundary_keeps_independent_closed_log_integral|Does the refined boundary proof enclose a known integral?|Verify closed logarithmic integrals in both integration directions and all sign orientations with pinned compiled Decimal arithmetic.
centered_boundary_restores_geometry_hooks|Does temporary proof refinement leave the producer unchanged?|Restore input and role callbacks after success or failure so subsequent calculations cannot inherit hidden state.
centered_gradient_refines_only_unit_auxiliary_partial|Can boundary-curve partials be tightened safely?|Recognize one constant unit auxiliary direction. Tighten only its constant coefficient and preserve every higher composed derivative.
""")

_add('packaging/test_opaque_tail_boundary','Independent bounded accuracy checking; complete evidence remains required.',"""
tail_direction_selection_and_complete_evidence|How does the tail checker choose a valid integration direction?|Use the twelfth transverse direction with radial fallback on known domain refusal, and radial only for the thirteenth. Keep complete intersections, reject disjoint or malformed evidence, and propagate cancellation and unknown failures.
tail_refuses_unqualified_piece|Can the experimental tail schedule affect unrelated pieces?|Reject every piece outside the two explicitly qualified tail indices, including incorrectly typed identifiers.
tail_keeps_independent_closed_integral|Does the selected direction preserve an independent answer?|Compare the scheduled complete boundary proof with closed logarithmic integrals in all sign orientations. Scheduling never changes acceptance.
centered_tail_requires_explicit_profile|Can a worker silently inherit a larger deadline?|Require the explicitly approved offline profile and reject a standard-profile mismatch.
centered_tail_preflight_routes_both_pieces|Are isolated trials wired to the intended checker?|Mock routing verifies both tail pieces use the64by8seed, material-only geometric inputs and the correct direction preference. Mock results are not numerical evidence.
""")

_add('packaging/test_opaque_centered_preflight','Independent bounded accuracy checking; complete evidence remains required.',"""
original_centered_tail_preflight|Does a whole tail piece meet its original accuracy share?|Verify the frozen source and input, then run one isolated complete-piece checker under the approved offline limits. Preserve failures, bound widths and charged work; this preflight cannot replace a fresh campaign receipt.
""")

_add('packaging/test_opaque_turn_diagnostic','Bounded independent accuracy evidence; no whole-M1 admission.',"""
saved_twelfth_turn_values_and_derivatives|Why do the last twelfth-piece regions refuse monotonicity?|Inspect three saved failed regions, both directions, line-wise angular values, first derivatives and curvature. Keep the20second/2048request diagnostic cap and distinguish broad intervals from actual turns.
""")

_add('packaging/test_opaque_centered_value','Bounded independent accuracy evidence; no whole-M1 admission.',"""
centered_values_reduce_dependency_and_keep_exact_polynomial|Can a narrower value enclosure retain a real turn?|Check a closed polynomial in either coordinate: value width shrinks, exact values remain enclosed, the derivative still includes both signs and higher coefficients remain unchanged.
centered_values_refuse_ambiguous_opt_in|Can value refinement be activated ambiguously?|Require a Boolean opt-in; the previous derivative-only path remains the default.
centered_values_reject_incompatible_point_value|Can conflicting value evidence be hidden?|Disjoint natural and mean-value inclusions raise an error rather than selecting an unsupported result.
value_centered_boundary_keeps_independent_closed_integral|Does value centering preserve a known complete integral?|Use independent logarithmic closed forms in both integration directions and all sign orientations.
""")

_add('packaging/test_opaque_value_regions','Bounded independent accuracy evidence; no whole-M1 admission.',"""
value_centering_on_saved_twelfth_turn_regions|Does value centering resolve the remaining saved refusals?|Compare complete direct and radial/transverse bounds on the three frozen worst regions. These local results do not admit a whole piece.
""")

_add('packaging/test_opaque_value_cost','Bounded independent accuracy evidence; no whole-M1 admission.',"""
value_centering_on_saved_expensive_regions|What does a stronger proof cost on previously expensive regions?|Compare directions on three saved expensive rectangles under the same diagnostic cap. An interrupted comparison remains failed; completed individual bounds cannot establish whole-piece speed.
""")

_add('packaging/test_opaque_value_tail','Bounded independent accuracy evidence; no whole-M1 admission.',"""
tail_direction_selection_and_complete_evidence|Does the radial-only schedule keep complete evidence?|Use only radial integration for the twelfth piece; retain the direct enclosure on known refusal, reject disjoint or malformed proofs and propagate unknown errors and cancellation.
tail_refuses_unqualified_piece|Can the new schedule alter another piece?|Reject every identifier except the explicitly qualified twelfth-piece index, including the separately passing thirteenth.
tail_keeps_independent_closed_integral|Does radial scheduling retain an independent exact answer?|Verify complete logarithmic integrals in all sign orientations. The local target schedules work and never replaces acceptance.
centered_tail_requires_explicit_profile|Can the new preflight silently raise its limits?|Require the approved offline profile and reject a standard-profile mismatch.
centered_tail_preflight_routes_both_pieces|Is the twelfth diagnostic wired to the intended proof?|Mock routing verifies the twelfth-only value-centred radial proof,64by8seed and material-only inputs. Mock outputs are not numerical evidence.
""")

_add('packaging/test_opaque_value_preflight','Bounded independent accuracy evidence; no whole-M1 admission.',"""
original_value_tail_preflight|Does the repaired twelfth checker meet its complete accuracy share?|Verify frozen input and source, then run one whole-piece diagnostic under unchanged offline caps. Preserve terminal failure or exact passing bounds; no isolated result replaces a fresh campaign receipt.
""")

_add('packaging/test_opaque_complementary_regions','Bounded work selection; complete numerical evidence remains mandatory.',"""
saved_remaining_twelfth_complementary_bounds|Which complete proof tightens the remaining broad regions?|Compare both value-centred directions on three saved largest regions under20seconds/2048requests. Preserve incomplete comparisons as failures; individual completed bounds do not admit a piece.
""")

_add('packaging/test_opaque_direction_precheck','Bounded work selection; complete numerical evidence remains mandatory.',"""
direction_precheck_uses_both_roles_and_only_relevant_knots|Can a cheap check avoid an unsuitable transverse attempt?|Inspect both geometric angular roles. Skip an expensive direction only as a work choice when a relevant derivative interval straddles zero; constant fields and absent knots remain distinct.
direction_precheck_propagates_cancel_and_unknown_geometry|Can work selection hide a stop or bad geometry?|Propagate cancellation and unknown geometry errors before any proof attempt.
direction_selection_keeps_complete_intersections|Does direction selection preserve the original correctness gate?|Intersect complete proofs, retain the radial fallback only for known domain refusal, reject disjoint evidence and propagate cancellation or unknown errors.
prechecked_boundary_keeps_independent_closed_integral|Does the selected proof enclose an independent known answer?|Exercise the selector and full boundary proof against closed logarithmic integrals in every sign orientation.
centered_tail_requires_explicit_profile|Can the selector silently change worker limits?|Require the explicit approved offline profile for the new trial mode.
centered_tail_preflight_routes_both_pieces|Is the new diagnostic restricted to the intended twelfth piece?|Mock the worker and verify the twelfth-only selector,64by8seed and material-only geometry. Mock outputs are routing evidence only.
""")

_add('packaging/test_opaque_prechecked_regions','Bounded work selection; complete numerical evidence remains mandatory.',"""
prechecked_direction_on_saved_remaining_regions|Do selected complete proofs finish the three-region diagnostic?|Run only the selected direction and known-domain fallback. Retain full bound intersections, direction choices, request counts and timing under the same diagnostic cap.
""")

_add('packaging/test_opaque_direction_preflight','Bounded work selection; complete numerical evidence remains mandatory.',"""
original_direction_tail_preflight|Does direction prechecking qualify the whole twelfth piece?|Verify frozen input and sources, then run a fresh whole-piece diagnostic with original accuracy shares and unchanged offline limits. Preserve every outcome; a preflight never substitutes for a fresh full-campaign receipt.
""")


_add('packaging/test_opaque_radial_resample','Diagnostic quadrature experiments; production admission is separate.',"""
radial_rows_keep_closed_polynomial_integrals|Does radial resampling preserve independently known integrals?|Reconstruct affine rows, integrate polynomial moments with closed answers, and require byte-exact identity at order5.
radial_rows_refuse_unqualified_inputs|Can malformed sampling data look like valid evidence?|Reject nonfinite, negative, nonaffine, incomplete and unsupported rows/orders.
radial_mapping_preserves_analytic_measure_and_linear_moment|Does concentrating samples change the integrated area?|Include each coordinate map Jacobian and compare constant/linear integrals with closed forms while retaining positive ordered nodes.
radial_mapping_rejects_unknown_and_nonaffine|Can an unknown map or nonlinear row bypass validation?|Require explicit rejection before diagnostic evaluation.
horizon_mapping_keeps_ordinary_and_selects_both_curved_sides|Are samples concentrated at the actual horizon on both sides?|Keep ordinary rows byte-identical and independently construct lower/upper affine measures to exercise opposite endpoint maps.
""")
_add('python/test_opaque_original_piece_error','Localization only, with the completed independent original-pixel reference.',"""
original_native_face_contributions_against_complete_reference|Which visible faces dominate the original rendering discrepancy?|Mask face contributions inside installed recomputation, reproduce saved complete colour/derivatives, then compare grouped certified pieces. Preserve the first disconnected-graph harness failure separately.
""")
_add('python/test_opaque_original_subpixel','Diagnostic image subdivision, not an adopted renderer repair.',"""
original_subpixel_refinement_against_frozen_reference|Does dividing the original pixel improve its accuracy?|Verify exact dyadic projection correspondence, sum serial subpixels and preserve mask; compare both reference endpoints. Failed2x2and4x4experiments remain evidence.
""")
_add('python/test_opaque_original_radial','Diagnostic direction-specific integration; no production promotion.',"""
original_radial_orders_and_point_material|Is the error in material evaluation or radial integration?|Reproduce installed five-node output, independently check three material points, then compare larger radial rules with both certified endpoints in bounded batches.
""")
_add('python/test_opaque_original_radial_maps','Diagnostic coordinate maps at unchanged sample count.',"""
original_endpoint_mapped_radial_rows|Can sample placement improve accuracy without adding nodes?|Compare squared and smooth endpoint maps with the unchanged independent reference, preserving every unsuccessful alternative and exact original baseline.
""")
_add('python/test_opaque_original_radial_horizon','Diagnostic horizon-aware sampling; native qualification remains required.',"""
original_horizon_mapped_radial_rows|Does concentration work on the correct side of the horizon?|Keep ordinary rows, select the appropriate endpoint for each curved side, include its Jacobian and compare original colour/albedo endpoints.
""")
_add('python/test_opaque_original_acceptance','Selected original fixed-geometry pixel across all13pieces, not whole-frame or moving-field admission.',"""
original_pixel_meets_independent_endpoints|Does the actual installed renderer meet the independently certified original image?|Verify the complete certificate identity and compare both endpoints for all three colour channels and constant-albedo diagonal derivatives at unchanged1e-10absolute+1e-9relative tolerance; require zero off-diagonal derivatives and existing GPU limits.
""")

_add('python/test_native_opaque_curved','Shared native bounded radial selection; unchanged800sample/32logical-block capacity.',"""
curved_radial_refinement_keeps_shared_capacity|Can spare-capacity refinement overflow crowded pixels or lose area?|Use one,two,three,six and seven disjoint dyadic conic triangles. Check fixed output allocation, dense-rule fallback, exact total area and zero whole-area coefficient derivatives; seven curves must refuse the unchanged logical-block ceiling.
""")

_add('packaging/test_opaque_translation_transport','Independent full-material first-derivative reference; not a forward-image or general-mesh certificate.',"""
translation_contains_closed_polynomial_and_parameter_directions|Does the transport identity include both field and moving-boundary terms?|Differentiate an independently known polynomial primitive and require its exact161 derivative inside the outward result.
internal_step_transport_is_not_the_zero_pointwise_derivative|Can a moving discontinuity contribute even when pointwise derivatives vanish?|Translate a unit step through a pixel and recover derivative minus1, which ordinary pointwise differentiation misses.
attached_visibility_edge_cancels_lower_transport_term|Does an attached moving edge carry the correct flux?|Compare with the closed primitive of an attached quadratic field and its moving lower endpoint.
interval_endpoint_uncertainty_is_preserved_with_negative_coefficients|Can sign changes silently discard reference uncertainty?|Use uncertain endpoints and negative transport speeds, checking the complete result enclosure.
invalid_transport_parameters_refuse|Can invalid endpoints or speeds produce accepted evidence?|Reject unordered or coincident endpoints, nonfinite speeds and Boolean inputs.
endpoint_shape_mismatch_refuses|Can mismatched channels look like a complete proof?|Require matching bounded endpoint channel counts before forming a derivative.
original_endpoint_intervals_contain_independent_decimal_algebra|Are endpoint radiances independent of the renderer?|Compare outward original-material enclosures with separate90-digit Decimal algebra at front and back points for normal and view fields; reject horizon endpoints.
endpoint_touching_uncertain_resource_knot_refuses|Can an ambiguous endpoint be assigned a one-sided derivative silently?|Refuse an endpoint overlapping a resource-knot enclosure before evaluating a branch.
""")
_add('python/test_opaque_translation_transport','Controlled full-original-material directions. Passing15-band diagnostics are not the default scene renderer or arbitrary-mesh admission.',"""
original_material_translation_derivative|Do moving normal, view and visibility fields include boundary transport?|Compare installed first VJPs with both endpoints of an independent Leibniz reference. Preserve default, curved-map and subdivision failures separately; keep800slots,16cuts and existing device/time limits. The factored harness contracts the native sample graph once per channel.
""")
_add('python/test_opaque_translation_scene','Actual public renderer camera-motion regression: opaque123/124 failures are retained; opaque125 finite-volume transport passes its independent controls.',"""
public_scene_camera_translation_has_independent_boundary_derivative|Can the public renderer give an accurate optimization direction near a horizon?|Translate the camera principal point over a fixed plane with the original material. Bound all radiance along both pixel edges independently, including the small vertical interval. Both camera axes and constant/vertex normals have independent endpoint bounds. Historical opaque123/124 runs fail; the fresh opaque125 repair passes. Forward bytes stay unchanged; device equality is checked separately.
""")
_add('packaging/test_opaque_curved_transport_map','Diagnostic coordinate maps only; no production sampling change.',"""
mapped_curved_rows_preserve_exact_area_moments_and_motion|Does a horizon coordinate map preserve measure and boundary derivatives?|Check exact area, first moments and moving subregion area for powers2,4,8 with complete Jacobians. Material-gradient diagnostics remain separate and may fail.
map_rejects_unqualified_shape_and_power|Can ordinary samples or unknown maps be mistaken for curved evidence?|Reject unsupported row shapes and powers before reconstruction.
""")
_add('packaging/test_opaque_streamed_motion','First-VJP harness qualification, not a renderer repair or second-derivative claim.',"""
streamed_motion_keeps_direct_and_sample_chain_rules|Can geometry backward work be collected without dropping material derivatives?|Compare combined direct and sample-chain terms with closed polynomial values and derivatives across three microbatch sizes, requiring exactly three sample VJP calls.
streamed_motion_handles_no_direct_parameter_path|Does the harness preserve a purely geometric derivative?|Exercise a field connected only through sample coordinates and require its exact derivative.
streamed_motion_requires_supported_bounded_profile|Can this helper evade established resource bounds?|Reject more than800sample rows or unsupported chunk sizes before evaluation.
""")

_add('python/test_opaque_composite_capacity','Retired opaque124 candidate1 only, not the installed production policy. The first fixture accidentally touched pixel boundaries; its failed derivatives remain historical.',"""
composite_plain_capacity_preserves_area_and_linear_moment|Does using spare sample capacity preserve exact geometric integrals?|Check occupied-slot limits, area, barycentric moments and interior triangle area gradients through32pieces;33candidates refuse. Use dyadic insets to avoid undeclared topology events. This cannot establish nonlinear material accuracy.
""")
_add('python/test_opaque_composite_probe','Explicit test-only remapping of a frozen retired wheel. No source-tree overlay, production promotion or relaxed numerical criteria.',"""
composite_scene_radial_diagnostic|Can a different bounded radial rule repair the public-scene gradient?|Compare eight symmetric radial rules with the same independent pixel-edge endpoints. All failed; their outputs and exact source snapshots remain evidence.
asymmetric_scene_radial_diagnostic|Can the same sample budget prioritize the front-facing side?|Reallocate fixed radial rows between the two sides, retaining full measure and gradients. All six tested allocations failed independent accuracy.
graded_scene_radial_diagnostic|Does concentrating intervals near the horizon resolve the remaining error?|Test six graded positive-side rules with validated Jacobians and unchanged total sample count. All failed; do not infer correctness from visual smoothness or device agreement.
""")
_add('python/test_opaque_composite_probe_controls','Closed-form validation of the diagnostic resampler; separate from the material acceptance tests.',"""
composite_probe_preserves_exact_area_and_horizon_flux|Does the diagnostic preserve both integration measure and moving-boundary flux?|Reconstruct analytic curved rows and require exact whole area, zero whole-area tangent and moving-subregion area/derivative, including opposite positive sides and graded intervals.
""")

_add('python/test_opaque_loss_transport','Scalar branch diagnostic, not a production pixel-gradient certificate.',"""
branchwise_loss_transport_matches_independent_endpoint|Are native matte-loss derivatives consistent across resource branches?|Integrate each branch derivative with Gauss2 in log cosine and compare both independent endpoint bounds, retaining the actual original resource and800-node cap.
""")
_add('python/test_opaque_boundary_transport','Native visible-edge nodes feed finite-volume camera gradients. Gauss12 is not a universal nonlinear accuracy proof.',"""
native_boundary_polynomial_line_integral|Do all four pixel edges integrate known polynomials correctly?|Compare degree0through6 line moments with exact1/(k+1), including barycentric reconstruction within floating roundoff.
native_boundary_depth_intersection_and_material_split|Which surface supplies colour along an edge?|Use an exact half-edge depth crossing and a separate material split; require the correct half-length per owner.
native_boundary_quadratic_split_and_empty_padding|Does a curved horizon split the edge without inventing coverage?|Check a quadratic root at one half, exact segment lengths and fully empty padding.
native_boundary_refuses_capacity_budget_and_input_derivatives|Can an unsupported call escape the bounds?|Refuse33candidates, insufficient workspace, invalid edges and differentiable boundary-node inputs.
native_boundary_rejects_edge_coincident_silhouette|Can an ambiguous silhouette derivative be silently chosen?|Refuse a triangle edge lying on the queried pixel edge.
native_boundary_exact_device_bytes|Do CPU and CUDA emit the same edge geometry?|Compare owner/sample bytes on all four edges with a material cut, using the CUDA worker resource profile.
native_boundary_c_abi_scratch_and_validation|Does the public C entry honor caller storage and validation?|Repeat with three scratch fill patterns, compare adapter bytes, and reject short workspace, aliases, bad indices and invalid edges.
native_boundary_diagnostic_header_precedes_edge_validation|Does an invalid diagnostic buffer remain untouched?|Supply a fully allocated but invalid diagnostic header together with a bad edge; require ABI mismatch without mutation. Preserve the original failed result.
native_boundary_refuses_unqualified_events_and_final_segment_overflow|Are tangencies and excessive edge subdivisions rejected?|Exercise tangent conics, edge-coincident material cuts and more than32final visible segments without truncating outputs.
native_boundary_degenerate_and_occluded_faces_do_not_contribute|Can invalid or hidden triangles add spurious flux?|Insert a degenerate face and a hidden surface; require only the nearest valid face to contribute unit edge length.
""")
_add('python/test_opaque_principal_transport','Production composition controls; independent gradient accuracy lives in the separate original-material scene tests.',"""
principal_transport_preserves_other_parameter_gradients|Does the camera repair disturb normal gradients or named cameras?|Compare ordinary normal derivatives and forward bytes with principal-point differentiation enabled/disabled, for direct and named cameras.
principal_transport_batch_loss_and_alpha_have_explicit_scaling|Do shared batches preserve normalization and background coverage?|Compare a single scene with a duplicated batch under mean loss, and require zero full-coverage camera derivative with background composition.
principal_transport_second_derivatives_refuse|Can an unsupported higher derivative look valid?|Request a differentiable backward graph and require explicit first-derivative-only refusal.
""")

_add('python/test_opaque_original_motion','Bounded actual original-mesh motion. Only exact transformation invariances have an independent acceptance reference; finite differences are screening, not proof.',"""
original_normal_input_vjp|Are tiny scalar device differences inside the renderer or in the caller reduction?|Compare all45018per-input normal Jacobian bytes before caller parameterization. Identical native arrays separate renderer parity from caller Torch reduction order; directional host sums remain diagnostic.
original_motion_diagnostic|Can the original mesh move without invalid gradients or unexpected refusal?|Check joint camera/mesh translation and normal scaling against exact zero derivatives. Screen a normal tilt and separate mesh/camera translations with two finite-difference steps. Preserve radial/translation overflow failures. Passing a diagnostic does not qualify arbitrary gradients.
""")
_add('python/test_opaque_motion_refusal','Diagnostic-only copied C11 kernel, never installed. Original sample bytes and refusal status are cross-checked against the installed wheel.',"""
original_motion_native_refusal|Why do small original-mesh movements refuse?|Record the native failed guard, occupied blocks and partial clipping fragments; compare independent rational visibility. The final diagnostic finds13exact pieces but34or36native logical blocks, exceeding32. Earlier instrumentation attribution and reference-coordinate mistakes remain historical. No tiny fragment is deleted or tolerance relaxed.
""")

_add('python/test_opaque_separation','Certified triangle separation skips only proved zero-overlap clipping. An uncertain predicate falls back to clipping; no arbitrary area epsilon is used.',"""
original_motion_separation_candidate|Does separation remove the original mesh capacity refusals without losing visible area?|Use a separately compiled copy of the shared kernel on four frozen original-mesh inputs; compare every visible face area with exact rational clipping. Shading accuracy is qualified separately through the installed wheel.
separation_proof_never_discards_positive_overlap|Can the separation predicate remove overlapping triangles?|Compare259 seeded and shared-edge cases with exact rational intersection. Include genuine overlap at2^-40 and require every certified separation to have exactly zero intersection area.
installed_separation_retains_tiny_positive_overlap_and_gradient|Are very small real contributions retained?|Require positive visible areas for triangle heights2^-20,2^-40,2^-60 and the analytic height derivative1/2 on the installed CPU/CUDA wheel. No epsilon deletion is permitted.
adapter_preserves_all_registered_dispatch_entries|Are all operation registrations retained?|Check all135 CPU/CUDA/composite dispatch entries from the preserved original adapter. Registration reachability is separate from numerical qualification of unexecuted operations.
""")


_add('python/test_opaque_original_leaf','Separate renderer input gradients from caller reductions and matrix-composition gradients. Device agreement is not an independent accuracy certificate.',"""
original_leaf_adjoints|Where do original-mesh scalar gradient differences begin?|Capture complete position, normal and camera-pose input adjoints for the original pixel. Compare separate CPU/CUDA runs and explicitly pairwise-reduced motion directions; preserve the raw bytes and distinguish caller reduction from renderer output.
original_camera_chain|Which camera calculation introduces a device difference?|Observe the streamed backward recomputation with hooks on the clip matrix and inverse pose. Equal incoming adjoints with different pose derivatives localize the inverse-backward discrepancy. Failed interception attempts remain historical.
""")
_add('python/test_camera_inverse_order','The inverse derivative is minus inverse-transpose times the incoming gradient times inverse-transpose. Fix its four-term accumulation order without changing the inverse forward calculation.',"""
saved_inverse_vjp_has_fixed_order_and_exact_rational_bound|Does the repaired inverse derivative retain accuracy and a reproducible accumulation order?|Replay the frozen original-scene inverse adjoints for float32/float64 and contiguous/strided gradients. Check every element against an independent exact rational triple product and require the declared pairwise bytes. Original failed tests remain preserved.
inverse_double_backward_preserves_existing_eager_support|Does the inverse repair preserve eager higher derivatives?|Run first- and second-derivative finite-difference checks on a well-conditioned non-diagonal matrix. This is a regression check, not the independent original-image gradient certificate.
""")


_add('packaging/test_opaque_tangent_flux','Independent integrated tangent corrections; a passing synthetic control does not certify the original mesh.',"""
moving_step_flux_sums_only_current_disjoint_leaves|Does a moving boundary contribute the right derivative exactly once?|Compare the moving-step integral with its exact negative unit derivative on three disjoint seed grids.
unresolved_flux_cannot_be_hidden_by_a_zero_smooth_derivative|Can unresolved boundary error hide behind a zero ordinary derivative?|Require refusal when the integrated correction interval is too wide.
tiny_nonzero_integrated_flux_is_retained|Are tiny real derivative contributions preserved?|Retain a positive 1e-70 integrated tangent without an epsilon cutoff.
wrong_flux_channel_count_refuses|Are malformed correction outputs rejected?|Require a refusal when correction and smooth channel counts disagree.
""")
_add('packaging/test_opaque_tangent_peano','Closed-form checks of ordinary parameter derivatives across angular table boundaries. Distributional parameter flux remains a separate required term.',"""
tangent_peano_contains_closed_piecewise_log_integral|Are angular-boundary residual values and derivatives enclosed independently?|Compare degree-one, two and three piecewise logarithmic integrals with closed 90-digit formulas in both axes and both crossing directions.
tangent_peano_preserves_zero_parameter_direction|Does a stationary field acquire a spurious derivative?|Require an exact zero tangent when the parameter does not change the field.
""")
_add('packaging/test_opaque_original_tangent','One strictly front-facing original visible piece, fixed-domain normal tilt only. Failed reference attempts remain unresolved evidence, not renderer failures or whole-pixel acceptance.',"""
radial_normal_tangent_matches_independent_decimal_vectors|Do the reference coordinates and normal derivatives describe the original mesh?|Compare all three cyclic triangle maps at four points with separately evaluated exact-rational interpolation and 90-digit vector normalization derivatives.
first_original_normal_tangent_certificate|Can the original normal derivative be independently certified within the standard limits?|Enclose values, ordinary derivatives and angular-table jump transport. Preserve cell/deadline refusals and complete-cover bounds; a partial enclosure cannot pass.
cyclic_original_normal_tangent_certificate|Does an equivalent coordinate map make the same independent certificate feasible?|Use the qualified cyclic map while retaining the same physical piece, normal direction, error criteria and resource caps.
saved_residual_regions_use_tighter_validated_tangent_bounds|Are narrower bounds supported by validated mathematics?|Compare integrated mixed residual bounds with conservative range enclosures on three saved difficult regions; overlapping intervals corroborate them without admitting the full piece.
cyclic_maps_retain_physical_triangle_and_tangent|Can changing integration coordinates change the represented object?|Require the same exact triangle vertices, area Jacobian and overlapping value/tangent enclosures at identical rational physical points; reject invalid anchors.
bounded_cyclic_map_comparison|Which equivalent map gives tighter complete-cover bounds at the same work budget?|Compare two maps with 63 cells and 30 seconds each, require their complete enclosures to overlap, and record tightness and cost separately from acceptance.
""")
_add('python/test_opaque_original_tangent','Installed-wheel comparison for one fixed visible original piece and one normal-tilt direction. No whole-pixel, moving-visibility, horizon or recomputed-normal admission.',"""
first_original_normal_gradient_meets_independent_endpoints|Does the actual original-material normal derivative meet the independent bound?|Pin the frozen input and qualified certificate, execute installed quadrature and Principled shading in bounded groups, and compare RGB and the first normal derivative with BOTH endpoints at unchanged accuracy. Check device bytes separately.
""")

_add('packaging/test_opaque_motion_inventory','Exact classification of the frozen original pixel before extending gradient evidence. This is topology evidence, not an integral or renderer accuracy certificate.',"""
original_motion_piece_inventory|Which original pieces need changing-horizon treatment?|Classify all thirteen exact visible pieces using polynomial sign certificates: nine strictly front-facing and four curved-horizon pieces. Preserve exact areas and local motion certificates.
""")
_add('packaging/test_opaque_motion_tangent','Independent checker extensions for normal, view and attached projected-geometry directions. Whole-pixel/world-camera/recomputed-normal gates remain separate.',"""
collection|Did the new motion-control test file load?|Preserve the initial syntax error as a collection failure; it is not a numerical result.
first_linear_log_join_keeps_ordinary_tangent_and_measure|Does the first angular-table join retain the full derivative?|Compare both crossing directions and a moving Jacobian with closed logarithmic value/tangent integrals.
curved_map_cannot_borrow_polynomial_crossing_count|Can curved crossings use an unsupported straight-triangle bound?|Require refusal when spatial monotonicity is unproved; no borrowed polynomial root count is accepted.
first_segment_retains_tiny_tangent_at_zero_cosine|Are horizon values and tiny real tangents retained?|Check the first linear segment at zero cosine and keep a positive 1e-70 tangent; equal-branch residuals remain exactly zero.
first_join_parameter_flux_encloses_known_discontinuous_loss|Does the first join include moving-boundary contributions?|Compare a known discontinuous loss with its exact parameter-flux bound and require zero for stationary fields.
original_motion_fields_match_independent_vector_chain|Are normal, view and geometry tangents represented correctly?|Compare exact rational interpolation and separately normalized 90-digit vector derivatives for all three declared directions.
original_curved_map_keeps_moving_horizon_zero|Does the moving root remain on the horizon?|Require exact zero view cosine and its tangent at the mapped boundary, with positive light cosine and preserved Jacobian behavior.
selected_original_motion_reference|Does this original piece and direction have a complete independent certificate?|Integrate ordinary derivatives, changing-domain measure and table-jump transport under the original caps. Preserve all depth, cell and deadline refusals. Partial enclosures are not acceptance.
curved_range_fallback_contains_closed_two_crossing_integrals|Can an uncertain crossing be bounded without assuming monotonicity?|Enclose closed polynomial value and ordinary-tangent integrals with conservative interval ranges; require separate parameter flux and retain explicit refusal without fallback.
unflattened_original_chart_preserves_angles_tangents_and_measure|Does changing coordinates alter the physical derivative?|Compare angles, parameter tangents and the complete change-of-variable Jacobian for normal, view and screen motion on an actual curved piece.
unflattened_flux_removes_coordinate_zero_without_discarding_transport|Can a harmless coordinate endpoint make a boundary bound fail?|Compare a fourth-power map and its unflattened chart on a moving step with exact unit flux; no contribution is discarded.
original_horizon_domain_preflight|Can the chosen coordinates satisfy the checker domain within existing limits?|Construct a complete segment-count cover with at most 256 cells and depth 10. This diagnostic does not certify integral or gradient accuracy; retain every refusal before considering a full trial.
saved_horizon_transport_chart_bounds|Does the targeted repair address the saved failed region?|Replay the exact first horizon-refusal box and retain both ordinary residual and parameter-flux bounds. This diagnostic is not whole-piece acceptance.
""")
_add('python/test_opaque_piece_motion_acceptance','Installed quadrature/shading comparison for one original visible face and its declared packed-input direction. Screen translation is not world-space vertex or camera motion.',"""
original_motion_meets_independent_piece_endpoints|Does the installed motion gradient meet an independent original-piece bound?|Pin input and certificate bytes, include the sample-coordinate derivative and direct shader derivative, compare BOTH endpoints, and measure existing CPU/CUDA resource limits. Device bytes are checked separately.
""")

GUIDES['tests/packaging/test_opaque_motion_tangent.py::collection'] = GUIDES.pop('tests/packaging/test_opaque_motion_tangent.py::test_collection')

_add('packaging/test_opaque_motion_subdivision','Independent subdivision bounds and domain-only diagnostics. None of these controls admits the original curved-piece integral or changes resource/accuracy limits.',"""
one_monotone_axis_uses_edges_not_unproved_corners|Can a proven direction reduce unnecessary subdivision?|Use exact polynomial extrema in both axis orientations and signs. Restrict the proved axis to an edge while retaining the full interval of the other axis; never infer extrema from unproved corners.
unproved_axes_preserve_interior_extremum_and_parameter_tangent|Can tightening discard an interior maximum or tiny real derivative?|Keep the closed interior maximum when neither spatial direction is monotone and retain a positive 1e-70 parameter derivative.
saved_segment_subdivision_diagnostic|Which direction reduces angular segments in the saved failed region?|Bound three saved parent regions and both pairs of children, recording actual angular counts and derivative-based hints. Preserve the initial rational-endpoint harness failure; this is diagnostic evidence only.
horizon_split_retains_exact_partition_and_ordinary_midpoints|Do graded cuts preserve the entire original region?|Check exact rational areas and both horizon orientations for the selected harmonic-distance cut. Earlier quarter/midpoint candidates remain in frozen reports; no cut removes a contribution or increases depth limits.
partial_monotonicity_requires_explicit_flag|Can the new bound change existing checker execution implicitly?|Require explicit Boolean opt-in with monotone bounds; existing callers retain their original evaluation contract.
unqualified_graded_integrator_is_refused_before_input_access|Can a diagnostic-only partition silently become an accuracy method?|Reject graded full integration before reading the scene until its quadrature and acceptance obligations are qualified.
""")

_add('packaging/test_opaque_knot_partition','Certified angular partitions, moving coordinate maps and independent integration controls. Domain coverage, numerical integration and installed-renderer gradient accuracy are separate gates.',"""
knot_cut_targets_both_axis_orientations_with_bounded_work|Can cuts target angular knots without unlimited search?|Check both coordinates and both orientations against exact affine roots, with at most14 recorded field evaluations. A target is guidance, not a root certificate.
knot_guidance_cannot_invent_root_or_swallow_stop|Can missing knots or cancellation cause false progress?|Return no cut for an absent crossing and propagate the cooperative stop before evaluating guidance.
exact_cut_integrates_full_measure_and_moving_jump|Does a nonmidpoint cut preserve area and the moving-boundary derivative?|Compare closed piecewise integrals and their parameter jump, in both axes and with a positive 2^-60 sliver. Require exact partition area and the original three-cell/depth-one cost.
invalid_cut_is_refused_without_budget_or_acceptance_bypass|Can an invalid cut corrupt coverage?|Reject an invalid axis, boundary/outside position or floating cut before subdivision; never turn the domain failure into acceptance.
valid_cut_still_obeys_existing_cell_and_depth_limits|Can smarter placement bypass a work limit?|Use a valid cut and require refusal when either the original cell or depth allowance is exhausted.
saved_original_knot_cut_diagnostics|Do knot targets help the saved original mesh regions?|Measure three parent bounds and both child bounds with exact area preservation and explicit guidance-call accounting; no integral or gradient acceptance follows.
cut_integrates_nonlinear_field_and_complete_moving_measure|Do curved shading terms and Jacobian motion survive partitioning?|Compare an exact polynomial spatial field, direct parameter derivative, moving measure and boundary-jump contribution with closed integrals in both axes.
curved_knot_root_and_motion_match_closed_implicit_derivatives|Can a knot-following boundary have independently checked derivatives?|Prove strict radial monotonicity and a root bracket, then compare both implicit derivatives with a closed nonlinear root in both orientations.
curved_knot_root_refuses_unproved_monotonicity_and_keeps_tiny_root|Can uncertain root topology or a tiny boundary be silently simplified?|Refuse an unproved monotone branch and retain the positive 2^-60 root and nonzero 1e-70 parameter effect.
saved_original_knot_contour_slices|Does the original angular knot follow a curved moving boundary?|Certify three fixed-transverse-coordinate roots and implicit derivatives on the saved failed region. Disjoint root intervals demonstrate variation; slices are not a whole curved strip or high-order integration chart.
knot_region_high_mixed_derivatives_match_closed_sqrt|Are the curved boundary derivatives used by quadrature independently checked?|Enclose all spatial coefficients through order10 and their first parameter derivatives against exact rational square-root coefficients in both orientations; retain an interval root, never an approximate point.
knot_region_uniform_bracket_and_boundary_refusals|Does the boundary exist throughout the region rather than just at sampled slices?|Require strict whole-box radial monotonicity and uniform endpoint signs. Refuse a midpoint-only root that touches/exits at an edge and propagate cooperative stopping.
knot_region_full_integrals_measure_and_shared_boundary_flux|Are moving area, shared edges and tiny curved regions preserved?|Compare closed full-domain integrals, unequal boundary traces and moving measure, including a positive2^-60 strip. Shared edges cancel only for equal traces; no tiny contribution is discarded.
knot_region_varying_physical_jacobian_and_jump_integral|Does the map include a physical Jacobian that varies in space and with the parameter?|Compare a polynomial physical measure, moving curved interface and unequal color traces against closed value and derivative antiderivatives.
saved_original_whole_knot_region|Can the original saved region be certified across its entire width?|Enclose a unique interior angular-knot root and order10 spatial/mixed derivatives over a complete transverse cover with bounded work. This local chart does not certify the complete mesh piece or its shading integral.
knot_band_proves_width_when_root_ranges_overlap|Can two moving knot boundaries define a positive strip when interval ranges overlap?|Prove radial monotonicity and use the mean value theorem to enclose the pointwise distance between ordered knots, in both orientations. Preserve mixed derivatives and refuse unordered knots.
knot_band_nonlinear_integral_preserves_parameter_conservation|Can the curved chart integrate nonlinear shading without corrupting gradients?|Compare polynomial, logarithmic and reciprocal integrals against closed antiderivatives. Include a nonlinear moving coordinate, its physical Jacobian and changing outer measure; boundary motion must cancel where the angular-domain integral is fixed.
original_interior_angular_band|Does the saved original region belong to certified shading branches, and can its actual material be integrated?|Prove adjacent view-knot boundaries and one uniform light branch, then optionally integrate the actual Principled specular and matte terms with complete moving measure. This local experiment does not qualify the complete piece or final renderer gradients.
curved_knot_cover_cannot_enable_unqualified_accuracy_trial|Can a successful domain-only curved cover silently enable an accuracy claim?|Reject the curved-knot option before reading scene input unless it is explicitly a domain preflight. Its later integration obligations remain separate.
original_curved_knot_domain_cover|Do certified curved cuts cover the complete original mesh piece within existing limits?|Try interior monotone knot cuts only when both child angular ranges meet the eight-segment cap. Charge both children, retain the complete-parent partition proof, and refuse depth/cell exhaustion without claiming accuracy.
uniform_knot_endpoints_use_proved_transverse_monotonicity|Can broad interval bounds falsely suggest an edge crossing?|Refine a boundary value only after proving monotonicity over its entire transverse interval. Check both radial orientations and retain the implicit parameter derivative; the legacy path stays unchanged.
uniform_knot_endpoint_refinement_cannot_hide_interior_crossing|Can endpoint samples hide an interior excursion?|Refuse an unproved crossing even when endpoint values alone look safe. Refinement requires explicit opt-in and cannot weaken topology acceptance.
saved_boundary_parent_targets|Is a different knot enough to repair the saved failed parent?|Diagnose all ten candidate knots on the exact depth-nine parent with bounded work; retain every refusal and do not turn a diagnostic into integral acceptance.
knot_edge_crossing_locations_and_parameter_motion|Can an actual knot entering or leaving a region be located rigorously?|Prove monotonicity in both coordinates and compare edge-crossing roots and parameter motion with closed quadratic values. Refuse folds, tangencies and uncertain endpoint contact.
saved_original_knot_edge_events|Where does the original knot enter the failed region and how does it move?|Certify the saved original knot70 crossing at the lower radial edge and its normal-tilt derivative. The event certificate does not yet construct or integrate a clipped moving-domain chart.
entry_chart_full_cover_and_moving_flux_have_closed_integrals|Does the entry-following map preserve the entire parent and moving-boundary contributions?|Integrate the pre-entry slab and both post-entry strips against closed area, first-moment and unequal-color formulas, including moving outer measure and a positive2^-60 scale. The collapsing entry edge is retained without discarding the interior wedge.
entry_chart_refuses_unsupported_topology_and_outside_chart|Can an unsupported entry pattern silently enter the integrator?|Require the certified single-lower-entry topology and unit chart coordinates; refuse other crossing patterns before claiming coverage.
entry_chart_nonlinear_inverse_root_integral|Does the chart work when the boundary is an implicit nonlinear root?|Compare the complete mapped piecewise integral and its boundary-motion derivative against closed square-root antiderivatives under bounded quadrature.
saved_original_entry_chart|Does the original region have a complete entry-following map with usable derivative bounds?|Verify all three charts on the saved parent in both quadrature directions, retaining order10 spatial and mixed physical-Jacobian coefficients. Certify each chart's angular domain without claiming full-piece shading accuracy.
entry_cover_cannot_enable_unqualified_accuracy|Can the new coverage path silently enable an original accuracy trial?|Refuse entry-following work unless explicitly selected as a curved domain preflight; complete numerical integration remains a separate gate.
original_entry_following_domain_cover|Can entry-following regions complete the original piece within the existing budget?|Charge each three-chart split as four binary-tree nodes with depths one/two below the parent. Independently preserve angular membership and refuse cell/depth exhaustion; partial coverage never qualifies accuracy.
original_knot_profile_requires_domain_qualification_before_trial|Can controlled generic integration silently admit the original horizon trial?|Refuse the original knot-targeted full accuracy profile until its complete domain cover is qualified.
original_angular_cover_cost_audit|Why did rectangular subdivision exhaust the budget?|Measure whole-radial monotonicity and outer-edge angular ranges on the original piece. This bounded audit motivates angular bands but does not itself accept a cover or an integral.
monotone_cover_closed_measure_jump_and_transport|Does a complete angular cover preserve moving measure and boundary jumps?|Compare full area, first moment and unequal-color integrals with independent closed formulas, including moving physical measure and a positive2^-60 scale. Include an angular band exiting through the outer edge.
monotone_cover_two_exits_and_closed_nonlinear_measure|Do adjacent bands remain complete when both boundaries leave the mesh?|Integrate a nonlinear inverse-root cover with two certified outer-edge exits. Require exact total area, first moment and zero spurious parameter motion.
monotone_cover_refuses_invalid_topology_budget_and_chart|Can the global band method bypass geometry or resource requirements?|Refuse unproved radial monotonicity, a nonzero horizon, unordered levels, uncertain endpoint contact, invalid chart coordinates and cell/depth exhaustion.
monotone_cover_retains_thin_positive_band|Can interval overlap erase a very narrow positive band?|Use the mean-value theorem to retain a2^-60 angular width and a strictly positive whole-chart integral enclosure. This range proof does not claim one-cell high-order quadrature resolves root-derivative cancellation; that failed attempt remains historical evidence.
complete_original_angular_cover|Does the entire original curved mesh piece fit the existing coverage allowance?|Prove global radial monotonicity, exact ordered level bands and all outer-edge exits. Validate every mapped chart with first physical-Jacobian motion, at most8angular segments,256charged cells and depth10. No shading or installed-gradient accuracy follows from coverage alone.
angular_cover_refuses_unqualified_accuracy|Can domain qualification silently enable a full accuracy trial?|Reject any attempt to use the coverage entry point as an accuracy method before reading input.
original_angular_chart_integration|Can the actual Principled material and normal derivative be integrated on a covered chart?|Run one local diagnostic with smooth terms, residual and conservative jump uncertainty. Charge the complete coverage tree and only use its remaining cell/depth budget. A local pass is not a complete-piece certificate; every refusal is preserved.
saved_angular_chart_stage_diagnostic|Which calculation prevents the saved original chart from integrating?|Inspect the same saved box at preparation, center evaluation, both high-derivative directions, residual and transport stages. Diagnostic completion is not numerical acceptance.
arb_sqrt_keeps_proved_positive_constant_and_mixed_coefficients|Can a series square root lose a proved positive denominator?|Check independent closed square-root coefficients through order10 and their first parameter derivatives. Refine only the series constant by intersecting an independently directed scalar bound; retain every higher coefficient.
saved_angular_chart_high_bounds_are_finite|Does the square-root repair unblock the original saved high-derivative calculation?|Require every original saved stage to evaluate after the constant-term repair. Finite bounds remove the arithmetic refusal but must still pass the separate integral error criterion.
centered_chart_factor_bounds|What still dominates the tighter matte error bound?|On the saved largest leaf, separately enclose view/light/half cosines, physical Jacobian and both angular loss factors through order10 and first parameter motion in both integration directions. Finite factor bounds are diagnostics, not an accepted integral.
angle_coordinate_closed_measure_and_transport|Does following the shading angle preserve moving area?|Compare the inverse-coordinate area, angle-weighted integral and first motion with exact closed formulas, including a 2^-60 angular scale. No renderer result is used as the reference.
angle_coordinate_nonlinear_inverse_high_mixed|Does the implicit inverse preserve high derivatives?|Compare coefficients through order 10 and first parameter motion in both chart directions against an independent closed quadratic inverse and Jacobian.
angle_coordinate_refuses_exit_and_nonpositive_slope|Can an unqualified angle map enter unsupported geometry?|The interior-only map refuses edge-exit charts and nonpositive analytic radial derivatives. The separate combined cover retains the qualified radial map at those boundaries.
original_radial_angle_derivative_matches_independent_seed|Is the original material's radial derivative correct?|Compare the analytic radial partial and its normal-motion derivative against independently seeded differentiation of the original normalized cosine.
angle_coordinate_original_factor_bounds|Does the angle coordinate remove the saved region's excess uncertainty?|Evaluate all saved high-order, mixed, residual and transport stages on the same hardest leaf. This is diagnostic evidence, not whole-piece acceptance.
angle_coordinate_original_chart_integration|Does the previously failing original chart meet its numerical criterion?|Fresh local Principled integration includes specular, matte, residual and motion bounds. Charge the complete cover tree and preserve all existing limits. A local pass does not qualify the whole piece.
angle_cover_complete_measure_jump_and_transport|Does the combined cover preserve total area and jump transport?|Compare exact complete area, first moment and discontinuous band integrals, including moving physical area and tiny positive scale, with no omitted edge regions.
angle_cover_two_exits_nonlinear_measure|Does the combined coordinate cover conserve a nonlinear moving domain?|Use a nonlinear field with both edge exits and an interior angle-coordinate band. Compare complete area, first moment and zero spurious motion against exact formulas under the same cell allowance. Preserve the failed all-angle predecessor.
complete_original_hybrid_angle_cover|Do all original regions retain a complete bounded map?|Inspect all 72 original charts with the interior angle map and radial edge/horizon maps. Require complete domain coverage, full moving Jacobians and the existing segment/cell/depth limits. No whole-piece integral is claimed.
angle_only_matches_all_saved_coefficients|Can inverse solving skip unrelated shading work without changing the answer?|Compare every interval endpoint through order10 and first motion on saved point, broad and near-horizon boxes using Decimal and Arb. Count avoided square roots; timings are observations, not an acceptance shortcut.
angle_only_refuses_unsupported_profile|Can the narrow angle helper silently handle an unqualified geometry?|Refuse noncurved or unsupported inside-horizon profiles before using the specialized unflattened outside map.
shared_charts_sum_complete_measure_and_transport|Do all regions share one complete integral and work budget?|Integrate a nonuniform binary chart tree against exact area, quadratic moment and first-motion formulas, including tiny positive measure. Charge internal tree nodes once and preserve inherited depth.
shared_charts_refuse_sum_of_local_passes|Can several passing local errors combine into an invalid complete result?|Construct individually acceptable derivative enclosures whose sum exceeds the global tolerance. Require the complete-cover integrator to refuse them.
shared_charts_never_reset_work_depth_or_deadline|Can starting the next region reset the limits?|Force global cell, inherited-depth and shared-checkpoint exhaustion across regions. Each must stop the single integration without admitting partial coverage.
shared_charts_validate_cover_before_work|Can malformed chart accounting enter the checker?|Reject incomplete binary-tree depth weights, invalid callbacks, excessive initial tree cost and ambiguous simultaneous fields before evaluation. Geometric disjointness remains the caller's separate proof obligation.
shared_charts_keep_hybrid_edges_jumps_and_tiny_measure|Does shared integration preserve moving edges and small contributions?|Integrate the complete angle/radial cover with unequal band colors and moving measure against exact closed area, moment and derivative formulas. Scale tolerance with the2^-60 area so smallness cannot mask error; retain the exploratory31-cell refusal.
shared_charts_refuse_missing_channel|Can a region silently drop a color or derivative channel?|Require every chart and evaluation to return the same channel count; refuse incomplete contributions before any acceptance decision.
original_hybrid_whole_chart_stage_preflight|Are horizon, interior and exiting-edge regions ready for a whole-piece trial?|Inspect the whole initial chart at preparation, center, high derivatives, residual and transport. Preserve every refused stage and distinguish completed diagnostics from numerical acceptance. No partial result or per-chart clock becomes a complete-piece certificate.
original_material_chart_state_and_frozen_stage_equivalence|Does combining the original regions change their material calculations?|Interleave preparation of horizon, interior and exiting-edge charts, require isolated extension/bounds/cache state, and compare center, high derivatives, residual and transport exactly against the hash-pinned earlier implementation.
original_material_shared_driver_keeps_construction_deadline|Can building the cover restart the reference clock?|Advance a controlled clock past90seconds during construction and require the shared original driver to stop before integration starts.
original_material_shared_driver_complete_piece|Does the complete original curved piece meet its value and gradient accuracy limits?|Build a fresh72-chart material cover and integrate every contribution with one summed enclosure,256cells,depth10 and90seconds including construction. Preserve the full refusal and partial trace if any limit stops it; partial coverage never qualifies the piece.
shared_original_initial_bound_diagnostic|Why did the shared whole-piece checker fail to finish?|Inspect the horizon, last completed region and previously passing local chart at full initial extent. Separate cosine, moving-Jacobian, angular-loss, residual and transport bounds in both directions. These finite bounds diagnose cost and uncertainty; they do not qualify the integral.
quadratic_sqrt_keeps_exact_square_and_moving_composition|Can a wide interval hide exact derivative cancellation?|Reduce quadratic square-root derivative numerators with exact rational algebra before bounding them. Compare against a known moving linear square root through order10 on Decimal and Arb; retain coordinate motion.
quadratic_sqrt_nonlinear_mixed_matches_point_algebra|Does the tighter square-root formula preserve nonlinear and mixed derivatives?|Compare against separate direct Taylor square-root algebra at three points through order10 and first motion, including nonlinear coordinate composition. Whole-interval bounds must overlap the point enclosures.
quadratic_sqrt_refuses_nonpositive_domain_and_keeps_tiny_curvature|Does the tighter formula hide a singularity or discard tiny curvature?|Refuse unsupported degree and nonpositive domains, and retain a positive second derivative from a2^-60 perturbation. No size threshold drops contributions.
tight_discriminant_preserves_closed_curved_area_and_motion|Does the repaired horizon map preserve moving coverage?|Integrate both complementary sides of a parabolic horizon and compare area and first boundary motion with independent closed formulas. Confirm the selected candidate actually runs.
tight_original_discriminant_point_and_motion_control|Does the repair agree with original-mesh point derivatives?|Compare all normalized input and Jacobian coefficients through order10 and first normal motion against the original algebra at three transverse points. This is separate from interval and integral acceptance.
tight_original_initial_bound_diagnostic|How much does exact cancellation tighten the difficult original regions?|Measure center, both derivative directions, separate residual and transport, and individual factors on original charts0/27/32. Preserve full reports; diagnostic completion alone does not accept an integral.
tight_original_chart_integral|Can the tighter bounds qualify the saved local integrals?|Fresh Principled value and normal-motion integration on charts0/27/32 includes residuals and jump uncertainty, with the entire cover tree charged to the same256cell/depth10 limits. A local pass is not a complete-piece certificate.
tight_shared_cover_matches_frozen_local_stages|Does the shared checker preserve the repaired local mathematics?|Compare center, radial and transverse order10 coefficients, residuals and jump transport exactly with the frozen repaired charts0/27/32. Interleave chart preparation to verify their mutable material state stays isolated.
tight_shared_driver_keeps_explicit_selection_and_construction_deadline|Can selecting the repair bypass the complete-piece deadline?|Verify explicit boolean selection, reject ambiguous values before work, and force a construction-time overrun under both selectors. Integration must never start after the same90-second deadline.
tight_shared_original_complete_piece|Does the repaired complete-piece checker meet combined accuracy and runtime?|Freshly build all72 original piece8 normal-motion regions and require one outward combined value-and-gradient enclosure within the original256cells/depth10/90seconds. Preserve incomplete or failed results without accepting partial coverage.
horizon_cache_exact_keys_ownership_backend_and_bound|Can bounded reuse change a proof result or leak mutable state?|Require exact interval endpoints for every stored coefficient, isolated returned Jets, distinct motion/order/backend/precision keys, a64entry eviction bound and no cached refusals. Reuse changes cost only, never the acceptance rule.
tight_shared_initial_cell_cost_diagnostic|What still costs time after exact horizon reuse?|Profile one fresh initial chart27 cell, retaining the expected one-cell accuracy refusal and complete local error components. Profiler timings identify cost only and do not qualify whole-piece runtime or accuracy.
prepared_angle_preserves_original_high_mixed_and_partial|Can repeated angle setup be reused without changing gradients?|Compare every coefficient through order10 and first normal, view and screen motion against the original field and analytic radial partial. Check broad boxes, both derivative directions, coordinate motion, ownership and the64entry bound.
prepared_angle_rejects_mismatched_jets_and_crosses_backend_safely|Can prepared data cross incompatible arithmetic contexts?|Refuse mismatched derivative orders and verify that changing Decimal to Arb cannot reuse results from the earlier arithmetic implementation.
prepared_value_only_brackets_match_exact_spatial_endpoints|Can root bracketing omit motion work it never consumes?|Require exact value and first spatial slope endpoints against the full original calculation. Only parameter-free order0/1 Newton brackets use this path; full motion coefficients and final Jacobians remain separate.
inverse_value_only_path_keeps_closed_mixed_derivatives|Does the cheaper root bracket preserve the complete inverse gradient?|Compare both integration directions through order10 and first motion with an independent closed nonlinear inverse and Jacobian. Confirm the restricted bracket path actually runs.
material_input_cache_keys_ownership_context_and_bound|Can repeated material inputs be reused safely across regions?|Keep at most64 field tuples across the entire piece. Require distinct chart, bounds, input coefficients, evaluator, backend and precision keys, fresh caller-owned outputs, exact endpoints, eviction and no cached failures.
material_input_cache_preserves_frozen_chart_after_prepare_and_interleave|Can material reuse mix regions or retain old local bounds?|Interleave original chart preparations, change the integration box, and compare cached results with fresh evaluation. Complete current bounds are part of cache identity.
prepared_spatial_high_order_is_exact_projection|Did the proposed high-order shortcut preserve exact bounds?|This rejected candidate attempted to omit unused motion arithmetic in high-order spatial work. Two exact-endpoint comparisons failed because sparsity changes backend operation selection. Preserve these failures; the candidate is absent from the accepted checker.
inverse_spatial_path_keeps_closed_mixed_derivatives|Did the rejected high-order shortcut pass its independent inverse control?|Compare with a closed nonlinear inverse through order10 and first motion. This control passing does not override the candidate's separate exact-endpoint failures; no full-piece trial used that candidate.
""")


_add("packaging/test_opaque_algebraic_angle_inverse", "Experimental independent inverse, transport and exiting-region bounds. Passing controls do not admit complete-piece or installed gradients; all terminal failures and limits remain authoritative.", """
residual_inverse_retains_closed_high_and_motion_coefficients|Does the cheaper inverse preserve high and mixed derivatives?|Compare both axes through order10 and first parameter motion against a closed quadratic inverse, retaining the exact moving Jacobian.
residual_inverse_refuses_uncertified_slope_and_bad_contract|Can an unproved inverse slope enter the checker?|Refuse an invalid residual callback and a nonpositive derivative before producing an inverse result.
original_quartic_residual_encloses_direct_vector_identity|Does the quartic equal the original normalized-vector equation?|Compare broad-box and point coefficients with independent normal/view dot-product algebra for normal, view and screen motion.
algebraic_shared_original_chart27_integrates|Does the cheaper inverse qualify an actual local material integral?|Integrate original chart27 with all correction and motion terms, charging the shared cover construction and inherited resource budget. A local pass is not whole-piece acceptance.
algebraic_shared_original_complete_piece|Does one fresh complete piece meet its combined accuracy and runtime gate?|Build all72regions and require a single summed enclosure under256cells,depth10 and90reference seconds. Preserve both terminal failures and do not accept partial coverage.
residual_cover_keeps_complete_moving_measure_and_tiny_terms|Can a cheaper inverse lose area or tiny contributions?|Compare complete area, first moment and motion with exact formulas, including2^-60physical measure and proportionally scaled tolerance.
algebraic_selector_is_explicit_and_keeps_construction_deadline|Can experimental options bypass selection or the shared deadline?|Reject invalid or incomplete selector combinations and force construction beyond90seconds before integration is allowed.
quartic_root_slope_identity_matches_direct_positive_derivative|Is the positive derivative identity valid at the chosen root?|Compare2*k*B*F_uwith a separately differentiated quartic at original normal, view and screen points. This identity is only a root derivative, not an arbitrary Newton slope.
saved_exit_transport_diagnostic_compares_both_inverses|Is the exiting-edge refusal new to the faster inverse?|Inspect the identical saved chart66box using both inverse paths. Diagnostic completion records failures; it does not certify an integral.
mapped_flux_keeps_collapsing_chart_jump_and_tiny_measure|Can a collapsing chart retain its moving discontinuity contribution?|Bound a closed geometric coarea integral where coordinate derivatives reach zero. Keep its known positive jump and a2^-60measure variant.
mapped_flux_refuses_missing_physical_monotonicity_or_measure|Can the geometric bound assume a missing proof?|Reject uncertified physical monotonicity or negative measure before producing a jump bound.
material_flux_override_preserves_both_roles_and_weights_measure_once|Does geometric transport count each material role and area factor once?|Check both angular roles and all RGB channels against a known7times measure, exact fallback behavior and invalid-bound rejection.
saved_exit_mapped_transport_retains_finite_complete_correction|Does the geometric bound unblock the saved original edge?|Require finite complete tangent corrections with both inverse paths. Finite bounds alone do not qualify the region or whole piece.
algebraic_mapped_original_chart66_integrates|Does fixing transversality also qualify the entire exiting region?|Freshly integrate all of chart66 within the inherited cell/depth budget and preserve its terminal accuracy refusal.
positive_exit_angle_keeps_closed_inverse_and_complete_mixed_measure|Does the positive exit-angle map preserve boundary motion?|Compare the inverse, moving transverse coordinate, angle and Jacobian with closed formulas through order10 and first motion in both axes; refuse horizon use.
positive_exit_angle_cover_integrates_all_moving_measure|Does the altered exit coordinate keep the complete domain?|Integrate the full mixed cover against exact area, first moment and motion, including a tiny physical measure without losing its positive contribution.
positive_exit_angle_original_chart66_integrates|Can the tighter exiting coordinate meet the original accuracy limit?|Freshly integrate the complete original region with inherited depth and all corrections. Preserve each failed attempt; improvement is not acceptance.
positive_exit_largest_leaf_error_diagnostic|What dominates the remaining exiting-region uncertainty?|Record both angular derivative directions, ordinary correction, fallback state and separate physical jump uncertainty on the saved largest leaf. This diagnostic does not certify a gradient.
normal_light_spatial_bound_matches_independent_normalization|Does exact cross-product cancellation retain the light derivative?|Compare both spatial derivatives with independent normalized-vector differentiation at multiple points and inside a complete interval box; refuse missing spatial jets.
normal_light_tangent_bound_keeps_intrinsic_and_coordinate_motion|Can cancellation erase intrinsic normal or coordinate motion?|Compare the complete parameter derivative against direct normalization, including2^-60motion and positive nonzero results.
exit_chart_forwards_current_residual_split_hint|Does adaptive subdivision use the right box's diagnostic hint?|Interleave cached boxes and require the current residual direction with unchanged correction values. Guidance chooses work order only; accuracy and global limits remain authoritative.
""")


_add("packaging/test_opaque_algebraic_angle_inverse", "Experimental correction bounds and independent inverse checks; no complete-piece admission without the full combined gate.", """
exit_correction_error_components|Which terms dominate the saved correction error?|Record ordinary and localized jump bounds in both directions for the two saved largest leaves; diagnostics alone do not accept an integral.
prepared_outer_edge_matches_direct_normalized_fields|Does direct outer-edge evaluation retain full motion?|Compare exact polynomial dot products against independent vector normalization through order10, broad intervals and tiny parameter motion.
repaired_exit_shared_original_complete_piece|Does the combined repair qualify the entire original piece?|Recompute all72regions and their combined error under one90second,256cell,depth10 budget. Preserve every timeout or incomplete enclosure.
repaired_piece_remaining_chart_diagnostic|Why did the next original region refuse a crossing bound?|Record both angular derivatives and the correction on the saved chart68box; a diagnostic pass is not accuracy acceptance.
repaired_initial_cell_cost_diagnostic|Where does the inverse calculation still spend time?|Profile one representative initial cell under fixed limits. Report profiler overhead and the full error enclosure; do not infer whole-piece performance.
repaired_exit_light_flux_is_finite|Can the shrinking light-angle region retain finite transport bounds?|Require a finite complete physical-coordinate transport bound on the saved refusal; tiny-measure and single-counted-weight controls remain separate.
incremental_quartic_inverse_matches_closed_mixed_coefficients|Does computing each inverse coefficient once preserve its mathematics?|Compare all mixed coefficients through order10 with an independent closed quadratic root, broad enclosures and2^-60motion; reject an unproved slope.
original_incremental_inverse_matches_preserved_recurrence|Is the new inverse connected correctly on actual mesh data?|Compare both axes, point and interval outputs with the preserved recurrence through order10; require the new solver to be invoked. Overlapping enclosures are a regression check, not an independent accuracy certificate.
positive_exit_angle_original_chart68_integrates|Does the newly exposed region meet complete local accuracy?|Recompute all local contributions within the inherited depth and remaining cell budget. This local pass cannot be combined with old local results to accept the piece.
exit68_largest_leaf_monotonicity_diagnostic|Is a broad light bound forcing a fallback?|Record complete-box derivative enclosures, fallback count and separately labelled point diagnostics on the saved largest leaf.
angle_light_differential_retains_coordinate_and_intrinsic_motion|Does eliminating the inverse coordinate preserve light derivatives?|Compare the exact-polynomial chain rule against independent normalized-vector derivatives, including intrinsic field motion, coordinate motion and2^-60terms.
exit68_light_differential_proves_saved_monotonicity|Does the repaired bound prove monotonicity over the saved box?|Require a strictly negative whole-box light derivative and zero broad range fallbacks; point samples alone cannot satisfy the test.
low_angle_inverse_bound_diagnostic|Does the squared inverse inflate low-angle error bounds?|Compare initial-cell enclosures with the preserved normalized recurrence; record uncertainty and cost without accepting an integral.
unsquared_inverse_matches_closed_mixed_coefficients|Does the unsquared recurrence retain all mixed derivatives?|Compare order10 and first-motion coefficients with an independent closed root, broad intervals and tiny nonzero motion.
unsquared_low_angle_bound_diagnostic|Does the unsquared recurrence improve the actual low-angle enclosures?|Record initial-cell error bounds for the saved original regions. Diagnostic success does not qualify a piece.
original_unsquared_inverse_matches_preserved_recurrence|Does the new recurrence match the preserved root equation?|Compare both spatial directions through order10 on actual original-mesh intervals and points.
unsquared_original_chart_integrates|Can the selected original region satisfy its full local error bound?|Compute fresh specular and matte value/motion enclosures with inherited depth and unchanged limits; no local-certificate sum may qualify a piece.
unsquared_shared_original_complete_piece|Does one fresh calculation qualify the whole original piece?|Combine all72regions under one256cell,depth10,90second budget, including construction, corrections and motion.
original_unsquared_residual_encloses_direct_vector_identity|Is the unsquared polynomial actually the original vector equation?|Compare direct normalized-vector algebra for normal, view and screen motion, points and boxes, and every mixed coefficient through order10.
unsquared_root_slope_identity_matches_direct_positive_derivative|Is the root-specific slope identity valid for the original fields?|Compare the identity with direct differentiation for three motions and positive angles; reject an uncertified slope.
unsquared_selector_requires_qualified_base_and_charges_construction|Can an experimental flag bypass prerequisites or the deadline?|Reject invalid selection and charge cover construction to the shared deadline.
physical_angle_inverse_keeps_closed_moving_radius_and_jacobian|Does solving for physical radius preserve the full derivative and measure?|Use an independent closed inverse and Jacobian through order10, both axes and2^-60motion; retain all nonzero contributions.
physical_angle_cover_retains_complete_moving_measure|Does the new map cover the whole moving region exactly?|Integrate a closed triangle area and first radial moment with exact motion, including entry and exit events and tiny total measure.
physical_original_map_matches_complete_material_jet|Do physical and normalized root routes agree on actual material derivatives?|Compare independent coordinate routes at matching targets through order10 and first motion, including positive entry and exit regions.
physical_selector_and_construction_deadline|Can physical coordinates bypass the declared proof chain or resource budget?|Require the explicit qualified inverse/exit selectors and test a construction deadline before any integration.
physical_original_chart_integrates|Does the physical-coordinate original region meet complete local accuracy?|Include specular and matte values, full first motion, correction and boundary transport under inherited depth and existing limits.
physical_shared_original_complete_piece|Does the physical-coordinate checker qualify all regions together?|Recompute the entire72region cover and combined RGB/basis error under one shared budget; preserve every refusal.
physical_entry_remaining_bound_diagnostic|Which terms dominate the saved entry-region uncertainty?|Record both integration directions, light/view/Jacobian coefficients and correction bounds. Label point values as diagnostics only.
physical_entry_light_bound_removes_range_fallback|Does exact light differentiation remove the broad saved fallback?|Require zero range fallbacks on the saved entry box while retaining all ordinary and boundary contributions.
physical_prepared_polynomial_keeps_outward_endpoints_and_context|Can cached coefficient conversion change any numerical bound?|Require exact endpoint equality against the original Horner evaluation, mixed-sign inputs, tiny terms and precision changes.
""")


_add("packaging/test_opaque_algebraic_angle_inverse", "Experimental reference optimization; unchanged numerical and resource gates apply.", """
physical_material_uses_direct_light_proof_without_repeated_cross_products|Can the physical path reuse its direct light proof?|Reject redundant per-sample cross-product reconstruction while independently verifying the direct differential, full material derivatives and complete local integrals.
""")


_add("packaging/test_opaque_algebraic_angle_inverse", "Diagnostic timing includes profiler overhead and is not whole-piece acceptance.", """
physical_remaining_cell_cost_diagnostic|Where does the remaining whole-piece work spend time?|Profile one saved dominant interior cell with the current physical map and preserve its complete error bound; use the result to target repairs, not to infer a full-piece pass.
""")


_add("packaging/test_opaque_local_smooth", "Independent checker proof and cost qualification. Local bounds and timings do not certify the complete piece; preserve every failure and all resource limits.", """
smooth_original_bound_diagnostic|Which direction makes the smooth bound loose?|Enclose order-ten derivatives on both axes of five saved dominant interior regions; compare bounded partitions without claiming an integral pass.
local_smooth_exact_polynomial_and_motion|Does the localized smooth proof enclose exact integrals and motion?|Use closed polynomial integrals through degree15, both axes, signed motion and2^-60contributions; require containment and positive tiny values.
local_smooth_kernel_masses_match_exact_total_and_symmetry|Are the localized Gaussian error weights complete?|Check outward truncated-power masses against the exact whole Gaussian remainder constant, symmetry, positivity and bounded partition contracts.
local_smooth_sharp_rational_has_independent_integral|Does the proof work beyond polynomials?|Compare a sharply varying rational integral and its moving-parameter derivative to independent logarithmic and rational closed forms in both directions.
local_smooth_explicit_contract_and_deadline|Can the candidate bypass proof or resource rules?|Reject malformed or unqualified selections and require checkpoint interruption during derivative work.
prepared_polynomial_reuses_transverse_rows_without_endpoint_change|Does repeated-work removal preserve every interval endpoint?|Compare to the original Horner evaluator for mixed signs, tiny terms and precision changes; check reuse and the64-entry cap.
local_smooth_original_comparison|Does a tighter bound repay its cost on the actual mesh?|Compare unchanged, four-interval and eight-interval proofs with fresh geometry on the saved interior or entry region, retaining all corrections and failures.
local_smooth_arb_exact_controls|Are the exact proof controls valid in the mesh checker arithmetic?|Repeat the polynomial, tiny-motion, kernel and rational controls using the pinned Arb series backend.
complete_cover_counts_all_contributors_without_local_acceptance|Does the report show every remaining contributor?|Count every current disjoint leaf while retaining only five detailed examples; label the whole-budget threshold as diagnostic rather than local acceptance.
transverse_reuse_controlled_cost|How much repeated calculation is removed?|Run an alternating cache-off/on/on/off same-cell comparison; require identical value and gradient interval endpoints and record all timings.
localized_chart_connection_and_no_implicit_budget_reset|Is the new proof connected only to its qualified chart class?|Verify72charts and143initial tree nodes, select positive full interiors only, preserve entry/exit paths and enforce the shared deadline.
localized_fresh_complete_original_piece|Does the complete curved piece now meet accuracy and runtime together?|Construct and integrate all72regions fresh under the shared256-cell,depth10,90-second reference budget; require combined value and derivative enclosures before acceptance.
""")


_add("packaging/test_opaque_tensor_smooth", "Independent checker qualification with unchanged accuracy and resource limits. Local passes do not imply a combined-piece certificate; every failure is preserved.", """
directional_tensor_remainder_diagnostic|Can we focus the smooth proof in the direction that dominates uncertainty?|Enclose both derivative directions and compare outward Gaussian Peano constants on saved original regions; no integral acceptance is inferred.
tensor_smooth_exact_integral_motion_and_tiny_terms|Does the directional rule contain known exact integrals and their gradients?|Check polynomial products through degrees8and15, signed parameter derivatives and2^-60contributions against closed integrals.
tensor_smooth_uses_24_samples_and_all_corrections|Does cheaper sampling keep every correction and nonzero term?|Count all24smooth samples and require exact integrals including tiny signed value and motion corrections.
tensor_smooth_rational_integral_and_moving_denominator|Does the proof work beyond polynomials?|Compare a nonlinear rational integral and changing-denominator gradient against independent logarithmic and rational answers.
tensor_smooth_contract_deadline_and_mixed_chart_cover|Can the rule bypass accounting or silently mix incompatible proofs?|Reject invalid selectors, charge a mixed24/25sample cover correctly and enforce interruption during rule construction.
tensor_gauss8_independent_moments_through_fifteen|Are the higher-order nodes and weights valid?|Require positive outward weights and exact polynomial moment containment through degree15.
tensor_smooth_compiled_exact_controls|Do independent controls pass in the actual mesh-checker arithmetic?|Repeat exact integral, signed motion, tiny contribution, moment, mixed-cover and deadline controls in the pinned Arb backend.
tensor_smooth_all_fifteen_original_regions|How much does the shared repair improve all15saved smooth contributors?|Rebuild geometry for every old/new one-cell comparison, alternate measurement order, retain all corrections and record full enclosures and elapsed times. Overlap is a consistency check, not independent accuracy proof.
tensor_smooth_complete_local_region|Can a difficult region meet accuracy with all boundary corrections included?|Build fresh geometry and adaptively integrate the entire selected interior or entry region, including moving measure and every residual term, under the original limits.
tensor_smooth_chart_connection_and_construction_deadline|Is the rule connected only to its qualified geometry family?|Check the72chart/143node cover, select positive fixed-upper-angle interiors and entries, retain exit/horizon rules and charge construction against the same deadline.
tensor_smooth_fresh_complete_original_piece|Does the complete curved piece satisfy combined accuracy and runtime?|Build and integrate every region fresh under one256cell/depth10/90second reference budget; require complete combined value and gradient bounds. Preserve a stopped trial as failed.
""")


_add("packaging/test_opaque_tight_transport", "Independent checker work under unchanged limits. Saved regions and speed measurements cannot replace a fresh complete-piece enclosure. Historical failed experiments remain visible.", """
remaining_combined_terms_diagnostic|Which correction terms actually dominate the remaining uncertainty?|Separate ordinary residual, spatial jump and parameter-flux bounds on six saved regions, and inspect smooth derivatives without claiming acceptance.
shared_correction_rule_contains_closed_piecewise_integrals|Can a shared quadrature rule preserve exact piecewise integrals?|Compare the experimental shared nodes with closed polynomial integrals including knots and signed corrections; preserve the rejected experiment.
shared_correction_keeps_closed_log_motion|Does shared sampling preserve a positive known logarithmic gradient?|Check exact logarithmic value and motion; central-node ambiguity failed some positivity checks, so this candidate was rejected.
shared_correction_reuses_exact_tensor_nodes_and_bounded_kernels|Are experimental shared nodes and kernel enclosures consistent?|Require exact node identities, bounded kernels and interval overlap; this checks the rejected experiment, not renderer accuracy.
shared_correction_original_saved_cells|Does shared sampling actually improve the difficult saved regions?|Measure full outward corrections; all six saved bounds worsened and the experimental shared correction rule was rejected.
ordinary_exact_zero_work_preserves_all_mixed_endpoints|Can we skip expressions without losing small gradients?|Remove work only for empty coefficient maps and compare every old/new endpoint, including tiny nonzeros and high-order terms.
light_differential_reuses_only_current_scalar_box|Can adjacent differential queries safely share scalar coefficients?|Require identical endpoints while changing arithmetic context, coordinate boxes and tiny motion; keep one immutable cache entry.
localized_tensor_contains_independent_exact_controls|Does the localized smooth remainder remain rigorous?|Repeat closed polynomial and rational integrals, signed motion, tiny terms, kernel moments and deadline checks in pinned Arb arithmetic.
localized_tensor_saved_smooth_regions|Do the remaining three smooth regions pass with corrections included?|Freshly integrate complete charts53,54and55 with the localized proof; these local passes do not qualify the combined piece.
localized_tensor_fresh_complete_piece|Does the entire curved piece meet the combined error and runtime gate?|Rebuild all geometry and every integral under one90second reference budget and unchanged256cell limit; retain stopped trials as failures.
redundant_work_controlled_cost_and_exact_piece_endpoints|Does less repeated work reduce actual local checking time?|Use an alternating old/new/new/old comparison with fresh geometry and exactly identical full local integral endpoints.
gradient_correction_selects_axis_without_changing_acceptance|Does subdivision follow the dominant gradient uncertainty?|Use a closed regression that passes only with the correct split direction; absent or malformed hints and exhausted budgets still fail.
tangent_centered_remainder_uses_zero_kernel_moment|Can a known exact cancellation tighten the correction proof?|Gauss5 integrates cubics exactly, so constant third derivatives add no remainder; retain tiny signed terms and reject inconsistent evidence.
tangent_centered_remainder_keeps_closed_log_motion|Does centering preserve gradients through material knots?|Repeat twelve closed logarithmic value/motion cases and an exactly zero direction, with every boundary jump still charged.
centered_tangent_original_saved_cells|Does zero-moment centering help the original correction bounds?|Compare all six saved correction intervals and require the centered result to stay inside its former outward enclosure; measure cost too.
direct_tangent_range_saved_cells|Could a direct interval bound replace costly correction work?|Compare natural range integrals with full Peano enclosures. The direct ranges were looser, so the shortcut was rejected.
fresh_refinement_schedule_checks_cover_and_charges_ancestors|Can reusing a work schedule accidentally omit a region or evade limits?|Require complete disjoint dyadic covers and correct inherited depths, reject gaps/overlaps, and charge every internal node plus leaf.
fresh_refinement_schedule_recomputes_all_fields_and_remainders|Is a seeded trial still a genuinely fresh accuracy check?|Count all new samples and derivative boxes, verify closed signed integrals and tiny terms, and reject insufficient cell budgets.
fresh_scheduled_complete_piece|Can a fresh trial finish faster by starting with known subdivisions?|Load only frozen geometry boxes, rebuild the physical cover, recompute all numerical evidence and apply the original combined error and resource gates.
""")


_add("packaging/test_opaque_tight_transport", "Independent checker experiments retain every failure. Explicit candidates require a fresh combined certificate under the original acceptance and resource limits.", """
full_light_differential_keeps_closed_mixed_coefficients|Does the higher-order light identity retain all mixed derivatives?|Compare orders0,3and10 against closed normalization formulas with coordinate and intrinsic motion and tiny terms; early scalar-dispatch test failures are preserved.
full_light_original_cells|Does the higher-order light identity repay its computation cost?|Compare six fresh original regions. The modest bound improvement did not justify the extra cost, so full trials leave this experiment disabled.
compiled_inverse_keeps_independent_mixed_roots|Does compiled inverse-series arithmetic still enclose the correct roots and gradients?|Check independent closed moving radii and Jacobians, through spatial order10 and first parameter order, including tiny motion and finite boxes.
compiled_inverse_keeps_complete_moving_measure|Does faster inversion preserve complete moving-region coverage?|Integrate independent closed areas and moments over full moving covers with every Jacobian and tiny contribution included.
compiled_original_cells|How does compiled inversion affect complete local bounds and runtime?|Measure paired fresh original-region enclosures and require interval consistency; the independent controls establish accuracy, not agreement alone.
fourth_tangent_keeps_independent_log_motion|Does the stronger correction proof preserve gradients at material knots?|Repeat closed logarithmic value and motion controls in both directions while retaining every jump and ordinary derivative term.
fourth_remainder_keeps_every_jump_order|Can higher-order error bounds accidentally discard discontinuities?|Check exact truncated-polynomial integrals through degree5 with both signs and tiny amplitudes, including jumps of orders0through3.
fourth_tangent_original_saved_cells|Does the stronger correction proof help the six saved problem regions?|Recompute both proofs, retain all corrections and report interval widths and elapsed time without treating a local result as whole-piece acceptance.
quarter_localized_smooth_keeps_independent_controls|Are finer smooth derivative domains still rigorous?|Repeat exact integrals, motion, tiny terms, sampling counts and deadlines. Earlier producers use paired maxima and half-kernel mass; the later qualified refinement uses separate outward quarter-kernel masses.
compiled_scheduled_complete_piece|Do the stronger bounds and faster calculations close the complete-piece gate together?|Build fresh geometry, use only a verified subdivision layout, recompute every contribution and apply the unchanged combined error, cell and runtime limits.
compiled_and_fourth_contract_preserves_refusal|Can experimental switches bypass prerequisite or arithmetic requirements?|Reject unsupported orders, implicit activation, uncertified roots or slopes, mismatched jets and missing pinned serial arithmetic context.
""")


_add("packaging/test_opaque_tight_transport", "Independent complete-piece checker qualification. Every small nonzero term remains in the final sum; local controls and diagnostics cannot close the combined gate.", """
conditioned_fourth_keeps_independent_controls|Does conditioning the correction on actual quadrature lines keep the proof valid?|Repeat independent truncated-polynomial and logarithmic-motion controls, retaining all jump orders and the positive tensor identity.
conditioned_fourth_original_saved_cells|How much does conditioning improve the six difficult saved corrections?|Measure both complete correction enclosures with every term retained; local overlap and tighter bounds do not establish whole-piece acceptance.
cached_arb_conversions_keep_exact_endpoints_and_context|Can repeated scalar conversions be reused safely?|Require identical final endpoints across precision changes, a4096entry cap, and complete cleanup after normal exit or interruption. No roots or integral results are cached.
conditioned_correction_keeps_samples_domains_and_deadline|Does the stronger correction proof preserve its work and stopping contract?|Count the same25samples, retain the complete ordinary proof and any needed conditioned domains, verify an exact tiny signed integral, reject unsupported selectors and propagate deadlines.
cached_conditioned_original_cost|Does scalar conversion reuse materially reduce original checking time?|Repeat both saved correction profiles with the optional cache. Exact endpoints are required; the measured gain is too small to solve the complete-piece deadline alone.
localized_kernel_mass_encloses_independent_exact_moments|Are the four smooth-kernel weights actually proven?|Bound absolute polynomial integrals using Bernstein coefficients on every subdomain and check independent Fubini moment identities. No equal-quarter-mass assumption is used.
weighted_quarter_smooth_keeps_independent_controls|Does the weighted smooth proof still contain known integrals and gradients?|Repeat polynomial and nonlinear rational controls, signed motion, tiny terms, sample counts, mixed covers and deadlines in the actual arithmetic backend.
all_scheduled_correction_ranges|Which corrections can use a cheaper complete proof?|Evaluate whole-domain value and ordinary-motion ranges on all93saved geometric leaves. Every returned enclosure is retained; this diagnostic makes no whole-piece acceptance claim.
conditioned_range_selection_retains_tiny_signed_enclosures|Does cheaper proof selection keep even extremely small contributions?|Check nonzero signed closed integrals, explicit selection of ordinary or conditioned proof, and deadline propagation. Width thresholds select a proof method, not acceptance.
conditioned_scheduled_complete_piece|Do the weighted smooth and selected correction proofs pass the whole-piece gate?|Rebuild the physical cover, recompute all93scheduled leaves and every later refinement, charge all ancestors, and require the same summed accuracy,90second and256cell limits.
""")


_add("packaging/test_opaque_tight_transport", "Qualified independent reference experiments. Local speed, a passing diagnostic or15local passes cannot replace the complete summed piece gate. All historical failures remain visible.", """
conditioned_proof_reuses_samples_and_escalates_only_if_needed|Can stronger proof work be limited without changing acceptance?|Count one25sample quadrature; use two ordinary proof domains first, then seven more only when needed, intersecting valid enclosures. The final piece still uses its original summed error rule.
certified_inverse_contraction_stop_keeps_closed_controls|Can the inverse stop redundant contraction without losing its root?|Retain the complete current certified root bracket and check independent moving roots, high mixed derivatives, Jacobians and full moving measures, including tiny terms.
inverse_contraction_compares_actual_whole_cells|How much does earlier contraction stopping save on actual cells?|Alternate enabled and disabled cases on four fresh full local evaluations, retaining all corrections and recording interval widths, Newton steps and runtime.
remaining_complete_calculation_profile|Where does the remaining whole-region calculation time go?|Profile one complete difficult region with every correction. Instrumented timing identifies cost; it is not a performance acceptance run.
fast_arb_conversion_keeps_independent_formulas_and_tiny_terms|Does cheaper number conversion preserve rigorous bounds?|Compare analytic and exact rational dual coefficients, moving roots and measures, signed intervals and1e-1000terms. Decimal text enters Arb outward; dyadic endpoints export with sign-aware directed rounding.
fast_conversion_complete_local_cost|Does repairing the profiled conversion overhead help actual checking?|Measure full original-cell calculations with the faster conversion, preserving every error term and independent accuracy controls. Local timing alone cannot admit the full piece.
selected_candidate_contracts_and_construction_deadline|Can the combined candidate bypass an explicit switch or its deadline?|Reject malformed switches, require the physical-angle prerequisites, and force expiry during construction before any integration is admitted.
selected_fresh_complete_piece|Does the fully qualified candidate meet combined accuracy and runtime?|Recompute the physical cover and every scheduled integral fresh with weighted smooth bounds, selected correction proofs and faster outward conversion. Apply the same basis/RGB error, cell, depth, memory and time limits.
selected_candidate_all_fifteen_smooth_regions|Do all15saved smooth regions work with the actual combined candidate?|Build fresh geometry for every region, include complete corrections, and record whether its one-cell enclosure meets the unchanged local error check. This is not whole-piece acceptance.
selected_proof_rules_in_fast_arithmetic|Do the proof components work together in the selected arithmetic?|Repeat independent weighted-integral, discontinuity, moving-measure, tiny-term and fresh-schedule controls together with faster outward conversion and certified inverse contraction.
""")


_add("packaging/test_opaque_tight_transport", "Independent mathematical checker evidence. Every stopped trial stays failed; local checks and speed improvements do not qualify the combined result. All numerical and resource limits remain unchanged.", """
reverse_conditioned_identity_keeps_independent_controls|Does reversing the integration identity retain every term?|Repeat closed polynomial and logarithmic motion controls in the transposed tensor identity, including unit and tiny signed amplitudes. The candidate retains its original direction because the comparison showed no overall advantage.
both_conditioned_directions_on_remaining_regions|Which direction bounds the actual remaining corrections better?|Evaluate both complete identities on three saved regions, record all enclosures and costs, and keep the independent accuracy gate separate from interval consistency.
repaired_selection_scheduled_complete_piece|Can the combined piece pass after repairing proof selection?|Recompute every coefficient and all101geometry-only leaves, charge100ancestors and further refinements, and enforce one summed value/gradient acceptance and one90second clock.
native_series_reuse_keeps_independent_controls_and_mutation|Can native arithmetic results be safely reused?|Check independent analytic coefficients, tiny motion, context cleanup and invalidation after coefficient replacement or mutation. The optional128entry cache remains disabled in the selected candidate.
native_series_complete_local_cost|Does native-series reuse save enough actual work?|Measure full local calculations with every correction. The measured gain was small, so this optional experiment is not selected for whole-piece trials.
partitioned_range_fallback_keeps_complete_measure_and_tiny_motion|Can a broad fallback be tightened without losing contributions?|Use four exact-measure proof domains with the parent extension fixed; verify closed unit/nonunit integrals, signed tiny motion, invalid contracts and interruption. No extra quadrature or independent budget is introduced.
partitioned_range_original_remaining_box|Does the partitioned fallback help the difficult actual region?|Compare whole and partitioned outward value/gradient ranges on the saved chart68box, including all corrections and cost. This local check is not whole-piece acceptance.
mixed_loss_polynomial_contains_independent_coefficients|Is combining the fixed material curves mathematically exact?|Generate known cubics independently, then verify all spatial/parameter coefficients through order10 and horizon terms with positive and negative tiny motion. Early harness float-division failures remain visible.
mixed_loss_keeps_actual_segment_jumps_and_tiny_motion|Does the cheaper material evaluation keep every knot contribution?|Check original-resource branch differences and ensure all one-sided segment enclosures and tiny derivatives survive. Exact polynomial controls establish the identity separately.
mixed_material_actual_complete_local_cost|Does one combined material polynomial reduce full local checking cost?|Use exact affine maps for four saved regions and recompute complete old/new integrals, corrections, gradients and times. Interval overlap alone is not the accuracy proof.
mixed_material_complete_piece|Does the exact material combination close combined accuracy and runtime?|Run a fresh whole-piece check under the same complete-cover and numerical/resource limits; retain any timeout as a failed trial.
mixed_material_contracts_and_construction_deadline|Can material combination bypass its prerequisites or deadline?|Reject malformed selectors and force expiration during construction in the explicit physical-angle candidate.
material_only_inputs_preserve_all_consumed_endpoints|Can correction-only queries skip the unused highlight field safely?|Compare every consumed cosine/Jacobian coefficient exactly on five actual chart classes at orders0/4/10. Full smooth shading continues to request the highlight field; no consumed parameter term is omitted.
material_only_actual_complete_local_cost|Does avoiding an unused field help complete local checks?|Compare both input paths with fresh geometry and all smooth, ordinary and boundary contributions, preserving any slower case too.
material_only_complete_piece|Does correction-only field evaluation satisfy the full-piece gate?|Recompute all scheduled regions fresh and require combined value/gradient bounds within the unchanged cell, depth, time, memory and thread limits.
targeted_schedule_validates_complete_depth_and_charges|Does targeted scheduling still cover the entire piece and count all work?|Verify the complete111leaf dyadic layout,110internal ancestors, inherited depth and existing rejection/field-recomputation controls. Only geometry is reused.
targeted_material_complete_piece|Can targeted fresh checking finally qualify the complete curved piece?|Recompute all111scheduled leaves and every further refinement with the qualified arithmetic/material candidate; require one complete combined enclosure and the original90second/256cell/depth10limits. This does not qualify other motions or the full pixel.
direct_quarter_conditioned_proof_keeps_independent_integrals|Can a stronger proof be selected directly without a preliminary bound?|Check exact polynomial and logarithmic integrals, signed tiny motion, all jump orders,25samples,nine proof domains and invalid selectors. The positive tensor identity is sufficient by itself.
direct_quarter_actual_complete_local_cost|Does the stronger proof reduce the dominant actual correction economically?|Compare complete saved-region bounds and time with both proof choices; preserve unchanged and slower cases and do not infer a whole-piece pass.

""")


_add('packaging/test_opaque_face_certificate', 'Independent checker qualification, not final renderer or milestone acceptance. Every historical failure and every nonzero contribution is retained.', """
face_sum_includes_every_signed_tiny_piece_outward|Does a face certificate include every visible contribution?|Sum all declared pieces outward in order, retaining signed tiny values and gradients. A piece is not a complete face.
face_sum_refuses_incomplete_mismatched_or_weak_evidence|Can incomplete evidence accidentally qualify a whole face?|Reject missing, duplicate, failed or mismatched pieces and nonfinite, reversed or overly wide intervals, including a sum that is too wide.
""")


_add('packaging/test_opaque_tangent_transverse', 'Independent checker qualification, not final renderer or milestone acceptance. Every historical failure and every nonzero contribution is retained.', """
transverse_tangent_preserves_closed_integrals|Does changing the boundary integration direction preserve the integral and gradient?|Check independent exact polynomial integrals through fourth degree in both tensor directions, reversed crossings and tiny signed motion. All jump orders remain included.
transverse_tangent_retains_unproved_crossings|What happens when crossing monotonicity is not proved?|Retain the original conservative enclosure. Failure to certify a faster method never removes a contribution.
exact_cosine_partials_enclose_independent_closed_derivatives|Are the tighter cosine slopes independently correct?|Compare exact-polynomial outward derivatives against independently derived normalization formulas with rational square-root brackets from integer arithmetic, on points and boxes.
exact_partial_refinement_activates_uncertain_crossings_and_refuses_disjoint_proof|Can an independent slope safely refine a broad crossing estimate?|Intersect the independent enclosure with the original slope, retain fallback without proof, and reject contradictory enclosures.
exact_partial_controls_crossing_count_before_localization|Does the proved slope govern crossing count and location early enough?|Supply a valid broad derivative and a tight independent slope; require root localization to consume the tight slope before deciding crossing count and orientation.
""")


_add('packaging/test_opaque_motion_extension', 'Independent checker qualification, not final renderer or milestone acceptance. Every historical failure and every nonzero contribution is retained.', """
requested_motion_reaches_geometry_before_construction|Does a requested motion actually reach the geometry constructor?|Intercept construction and verify piece, motion and cyclic anchor before any numerical work.
angular_selection_refuses_undeclared_inputs_before_geometry|Can an invalid selector silently become another motion?|Reject unsupported piece, motion and anchor values, including booleans used as integers.
complete_and_failed_reports_keep_requested_identity|Do success and failure reports identify exactly what was checked?|Verify piece, face, motion and anchor propagation; a failed reference cannot be mistaken for a different successful case.
declared_motion_builds_fresh_complete_cover_and_keeps_horizon|Does each motion preserve complete moving geometry?|Build the full original curved cover, verify exact domain measure and independent vector chains, and retain the moving horizon zero.
selected_arithmetic_retains_independent_roots_measure_and_tiny_motion|Does faster arithmetic preserve the independent moving-map proof?|Repeat closed root, full-measure and tiny signed-motion controls in the selected outward arithmetic.
selected_profile_refuses_a_noncurved_piece|Can a curved-only proof be applied to unrelated geometry?|Reject the incompatible piece before claiming a curved profile result.
selected_motion_piece_fresh_reference|Does the entire curved piece pass for this exact motion?|Recompute all terms from frozen input and a geometry-only schedule; require combined value/gradient bounds and unchanged time, depth and cell caps.
front_correction_selection_refuses_invalid_profiles|Are experimental correction choices explicit and supported?|Reject invalid orders and missing prerequisites before geometry or integration.
front_selected_correction_keeps_all_constructor_obligations|Does the front reference retain all correction obligations?|Verify fourth-order conditioning, mixed material, centered bounds, horizon handling and front-only scope at construction.
front_saved_correction_comparison|Does a stronger proof help the saved front-piece regions?|Measure old and new complete enclosures on three saved boxes. Interval overlap is consistency, not independent accuracy acceptance.
front_schedule_forwards_complete_single_chart_and_charges|Can a saved layout bypass full-domain work accounting?|Verify the complete chart, all correction callbacks and the same cell/depth limits reach the shared integrator; no numerical values are reused.
front_selected_piece_reference|Does the scheduled front piece pass as a complete sum?|Rebuild every field and correction on the full scheduled domain and enforce unchanged combined accuracy and resource limits; preserve the stopped trial.
front_transverse_profile_requires_the_qualified_correction|Can transverse integration activate without its prerequisites?|Reject implicit or incompatible switches before construction.
front_transverse_constructor_preserves_selected_profile|Does adding crossing integration keep the existing proof profile?|Verify the selected correction constructor still receives all required horizon, conditioning and material obligations.
front_transverse_saved_comparison|Does integrated crossing treatment improve the actual saved regions?|Recompute complete local enclosures with and without the optional proof; record costs and retain unsuccessful experiments.
front_remaining_proof_components|Which mathematical bound is still dominating uncertainty?|Record ordinary and jump terms in both directions on a saved region, without treating a diagnostic as acceptance.
front_conditioned_directions_saved_comparison|Which integration direction is useful for this front piece?|Compare both controlled tensor identities on the same three boxes, including every jump and transport term, with separate runtime measurements.
front_selected_gradient_guidance_reaches_shared_integrator|Does the checker subdivide according to the dominant gradient uncertainty?|Verify that the selected front profile passes its gradient-error direction to the shared integrator. Acceptance remains the full outward sum.
front_axis_selection_requires_conditioned_profile|Is the integration direction explicitly bound to the qualified profile?|Reject unsupported directions or missing conditioning and verify the selected axis and independent slope proof reach the constructor.
front_guided_piece_reference|Does the repaired checker qualify the complete front piece within the original limits?|Start a fresh complete domain, guide subdivisions by gradient uncertainty, retain every smooth and boundary term, and stop at the unchanged error, time or cell limits.
""")


_add("packaging/test_opaque_motion_extension", "Independent checker diagnostics. Local improvement cannot substitute for complete summed accuracy, and every failed report remains available.", """
front_exact_partials_saved_boxes|Do exact parameter polynomials reduce actual angle-derivative uncertainty?|Measure the original and exact rational-polynomial bounds on three saved regions. The method remains a diagnostic; an earlier endpoint-construction error is preserved.
front_material_only_keeps_all_consumed_endpoints|Can an unused half-vector calculation be removed safely?|Across three motions and orders0,1,4,10 require exact endpoint equality for both cosines and the Jacobian, including every parameter coefficient. The smooth shader still computes its half-vector.
front_exact_motion_actual_cost|Does the exact motion derivative materially improve the complete bound?|Compare complete corrections and runtime on three saved regions. Improvement below0.02percent at added cost leaves the experiment disabled; no full trial is justified by this diagnostic.
front_material_only_actual_cost|Does omitting unused work reduce complete local cost without changing results?|Compare three full local value/gradient enclosures with and without the explicit material-only query. Saved rows are exactly identical; local runtime is not whole-piece qualification.
""")


_add("packaging/test_opaque_tangent_transverse", "Independent motion-chain proof, separate from complete-piece performance and acceptance.", """
exact_cosine_motion_keeps_intrinsic_and_coordinate_chain|Does the exact cosine motion retain every chain-rule term?|Check intrinsic normal motion together with both moving coordinates and tiny positive/negative terms against independent normalized-vector formulas.
""")


_add("packaging/test_opaque_boundary_flux", "Independent boundary-bound controls and saved-region diagnostics. Only the separately frozen complete-piece certificate admits that piece; no installed face/pixel acceptance is inferred.", """
saved_boundary_components|Which boundary contribution dominates the checker uncertainty?|Separate true distributional brightness-jump flux from the ordinary tangent correction on exactly three saved regions; retain both contributions.
saved_weighted_jump_components|Where do conservative crossing bounds become too wide?|Record root support, varying coefficients, signed kernel mass and both tensor directions without treating the diagnostic as acceptance.
weighted_mass_uses_exact_first_segment_and_retains_tiny_terms|Does absolute kernel mass remain rigorous for tiny contributions?|Use exact first-segment integrals for orders1to4 and both signs at2^-80 scale; retain every nonzero term.
weighted_mass_contains_independent_closed_polynomial|Does the new bound contain independent analytical integrals?|Integrate each polynomial kernel segment independently at high precision, with both signs, several supports and all four orders.
weighted_mass_keeps_uncertain_roots_and_refuses_invalid_support|Can uncertain endpoints or invalid domains escape the proof?|Keep the full root interval, test enclosed endpoint choices, preserve exact zero-length support and refuse invalid domains.
weighted_mass_selected_arithmetic_and_bounded_cache|Does the selected arithmetic preserve proof and resource obligations?|Repeat the independent controls in the selected backend; check immutable returned intervals and the fixed128-entry cache cap.
saved_weighted_mass_comparison|How much do complete bounds improve on the three saved regions?|Compare whole local corrections plus distributional flux before and after the mass repair. Local timing is diagnostic and sensitive to machine load.
saved_boundary_cost_profile|Which calculations and scheduling delays remain costly?|Record CPU time, wall time and instrumented call costs on the same regions. Profiling is not a whole-piece performance claim.
quadrature_lines_skip_unused_continuous_proofs|Can unused continuous proofs be omitted without changing the result?|Check both positive tensor identities against a closed polynomial integral; retain five line bounds and four continuous strips with exact endpoint equality.
actual_front_builds_continuous_proofs_only_on_consumed_strips|Does the real front-piece checker use that cost repair?|Inspect the original saved region and require continuous proofs only on the four strips that consume them, retaining all line derivative and jump bounds.
saved_line_proof_endpoints_and_cost|Does the cost repair preserve actual saved-region results?|Recompute all three regions with both routes and require exact value/tangent endpoint equality; record CPU and wall times separately.
""")


_add('packaging/test_opaque_motion_progress', 'Independent numerical checker controls and bounded saved-region diagnostics. Local gains are not complete face/pixel, installed renderer, world geometry or M1 acceptance. Every historical failure and tiny contribution remains.', """
remaining_front_piece_proof_directions|Which integration direction helps the next visible piece?|Recompute three saved regions in both directions, capturing complete correction components and elapsed time; preserve every jump and motion term.
crossing_contraction_contains_all_independent_nonlinear_roots|Can tighter root brackets accidentally omit a shading crossing?|Compare interval-Newton brackets against independent integer-square-root bounds for nonlinear fields, both orientations and uncertain transverse coordinates.
crossing_contraction_keeps_tiny_roots_uncertainty_and_cancellation|Do small roots and unresolved slopes remain safe?|Retain a nonzero root smaller than2^-70, leave a bracket with an uncertain derivative intact, certify absence only through enclosure, and propagate cancellation.
crossing_contraction_selected_backend|Does compiled outward arithmetic preserve the crossing controls?|Run the independent nonlinear, reversed, tiny-root and cancellation checks in the explicitly selected ball backend.
discrete_line_proof_contracts_before_forming_jumps|Are refined crossings actually used in the line calculation?|Inspect the selected correction and require contraction before all jump orders are formed; no line derivative is omitted.
continuous_proof_contracts_only_certified_interior_edge_roots|Can an absent or clipped crossing be treated as an interior root?|Require contraction only when endpoints independently prove a strict interior crossing; retain all other endpoint possibilities.
exact_slope_ratio_cancels_shared_norm_and_retains_full_box|Can a common positive normalization be cancelled rigorously?|Check the rational derivative ratio against independently closed formulas on points and complete boxes, preserving outward bounds.
transverse_proof_uses_independent_exact_ratio|Does the real boundary proof consume the exact ratio?|Inspect the callback and require its full-box ratio in the continuous boundary integration, rather than merely computing an unused diagnostic.
remaining_front_piece_bound_components|Which saved-region contribution is still too uncertain?|Record ordinary and jump terms separately in both directions; diagnostic widths do not establish complete-piece acceptance.
remaining_front_cyclic_maps_preserve_physical_fields_and_motion|Does changing the triangle coordinate anchor change the problem?|Compare physical fields, parameter derivatives and measure under all three cyclic anchors, for both neighboring pieces and all three declared motions.
remaining_front_coordinate_feasibility|Which coordinate map has independently strict slopes?|Count certified spatial slope signs on16boxes for each cyclic anchor; this selects work and never accepts an integral.
remaining_front_subdivision_response|Which subdivision tightens the next piece economically?|Compare parent and both possible child partitions on three saved regions. Residual centers from different smooth extensions are not interchangeable; only uncertainty guides work.
contraction_reuses_independent_slope_without_rebuilding_field_jets|Can an already qualified derivative avoid redundant field calculation?|Supply an independently exact full-bracket slope, retain midpoint value queries and require no order1 field reconstruction while enclosing the known root.
independent_slope_reuse_keeps_uncertain_roots_and_full_bracket|Does the cheaper root calculation retain every possible crossing?|Verify reversed nonlinear roots using independent integer arithmetic, full transverse uncertainty, a tiny positive root and the unchanged unknown-slope fallback.
next_piece_saved_transport_and_contraction_cost|Does the cost repair preserve the next piece's actual local results?|Recompute three saved piece3 regions with the frozen former and current contraction, require exact ordinary and combined-tangent endpoints, and record both local times.
next_piece_subdivision_response|Does radial refinement help the third visible piece too?|Compare both partition directions on its three dominant saved regions, retaining each child's full correction; local uncertainty and timings are only scheduling evidence.
""")

_add('packaging/test_opaque_face_certificate', 'Artifact and outward complete-face composition controls. Producer and worker evidence is audited separately; no whole-pixel or world-geometry acceptance.', """
face_bundle_checks_input_artifact_bytes_and_recomputed_sum|Can a face comparison accept altered or incomplete evidence?|Check bundle and certificate bytes, input identity and exact recomputed sums; reject altered hashes, missing or duplicate pieces, failed certificates, changed directions and forged sums.
""")

_add('python/test_opaque_piece_motion_acceptance', 'Installed float64 complete visible-face comparison at the selected original pixel, three declared packed-input directions. No whole-pixel, world geometry, camera pose, recomputed-normal or final showcase admission.', """
original_motion_meets_independent_complete_face|Does the installed renderer match every contribution of this visible face?|Verify frozen reference artifacts, recompute the outward sum of all saved pieces, compare RGB and gradients to both endpoints at the unchanged tolerance, and enforce GPU storage. CPU/CUDA bytes are compared by the campaign audit.
""")


_add('packaging/test_opaque_flat_resample', 'Research-only flat quadrature controls; these do not admit native rendering or a whole original pixel.', """
flat_refinement_preserves_closed_triangle_moments_and_motion|Does denser sampling preserve the same surface integral?|Compare triangle moments and their area-motion derivatives to closed rational formulas at several orders.
flat_refinement_refuses_unqualified_rows_or_capacity|Can research refinement bypass a native work bound?|Reject nonaffine, nonfinite or excessive-capacity input before producing samples; keep all existing fan contributions.
""")
_add('packaging/test_ptx_resource', 'Windows build-resource byte controls; loading and runtime behaviour are separately qualified on the fully rebuilt installed wheel.', """
ptx_resource_preserves_every_input_byte_and_exact_terminator|Can the memory-efficient CUDA packaging change the program bytes?|Generate payloads with arbitrary binary data and require every input byte followed by exactly one final NUL and a matching resource-size declaration.
ptx_resource_refuses_empty_payload_and_unsafe_resource_identifier|Can an invalid resource masquerade as a usable CUDA program?|Reject empty input and resource identifiers outside the supported integer range.
""")
_add('python/test_native_opaque_flat', 'Shared installed CPU/CUDA capacity-counted flat quadrature: closed polynomial checks and exact sample/first-VJP bytes. Nonlinear shading, complete faces and original-pixel motion remain separate gates.', """
flat_capacity_keeps_closed_moments_and_area_motion|Does the denser native flat rule retain the correct area and geometry derivative?|Compare fourth-order moments and area motion to independent closed formulas while counting actual samples under the unchanged 800-slot limit.
flat_refinement_preserves_curved_priority_and_combined_capacity|Does improving flat sampling steal capacity from curved pieces?|Check mixed layouts against the fixed curved priority, actual sample counts and unchanged combined capacity.
flat_refinement_retains_independent_face_and_logical_block_refusals|Can a denser fan evade a face or block limit?|Require the existing independent 32-face and 32-logical-block refusals, with no silent discarded contribution.
flat_shared_cpu_cuda_sample_and_first_vjp_bytes|Do the two devices sample and differentiate exactly the same rule?|For both precisions, compare owners, samples and every first input VJP byte; measure GPU allocation against the existing allowance.
""")
_add('python/test_opaque_piece_motion_acceptance', 'Research-only installed tensor refinement; complete-face/native admission requires its separate frozen campaign.', """
complete_face_flat_tensor_diagnostic|Does a denser flat rule improve the saved nonlinear face?|Preserve all visible fans, then compare its colour and declared motion derivative to both independent outward reference endpoints. This diagnostic is not native or whole-pixel qualification.
""")


_add('packaging/test_opaque_motion_progress', 'Saved local correction comparison; widths and cost guide work but cannot qualify a complete piece or motion.', """
piece10_saved_correction_directions_and_refinement|Which direction makes the next front-piece proof tighter without losing terms?|Verify the stopped trial bytes, recompute its three largest tangent-error regions in both integration directions, include both radial children, and report complete correction bounds and elapsed time. No residual center is promoted to a result.
""")

_add('packaging/test_opaque_motion_extension', 'Fresh independent original outside-horizon piece9 reference; no copied piece8 certificate, seed schedule or installed admission.', """
remaining_outside_curved_fresh_reference|Does the remaining outside-horizon piece have a complete independently bounded motion integral?|Build its own angular cover under one construction-inclusive clock, retain all ordinary and boundary terms, and require the unchanged uncertainty and cell/depth limits. A failed or incomplete cover remains unresolved.
""")

_add('packaging/test_opaque_motion_extension', 'Bounded local proof comparison; no complete-piece, production, device or milestone admission.', """
piece9_saved_transverse_correction|Can certified integration across a shading boundary tighten the curved pieceÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢s remaining error?|Pin its stopped trial, rebuild its own angular cover and compare all correction terms on its three largest saved regions, alternating the local timing order. Retain every jump and parameter flux; local gains require a fresh complete trial.
""")

_add('packaging/test_opaque_tangent_transverse', 'Independent closed curved-boundary integrals and explicit proof-profile refusals; saved original-piece and complete-pixel qualification remain separate.', """
curved_crossing_transverse_integral_has_closed_polynomial_reference|Does transverse boundary integration remain correct when the boundary is curved?|Compare both integration directions and orientations for positive-part polynomials separated by a parabolic boundary, including a negative tiny tangent. Exact rational double integrals are independent of the numerical kernel.
curved_transverse_profile_is_explicit_and_preserves_monotonicity_refusal|Can the new curved proof silently accept an unproved crossing?|Require an explicit fourth-order curved/transverse profile and reject a spatial derivative whose sign is not certified before locating crossings; preserve the prior unselected guard.
""")

_add('packaging/test_opaque_motion_extension', 'Explicit curved-reference profile binding and refusal controls. They do not admit a complete numerical integral, installed renderer or milestone.', """
curved_transverse_profile_refuses_unqualified_cover_before_geometry|Can the curved-boundary proof activate without its qualified prerequisites?|Reject ambiguous flags and missing physical-angle, conditioned fourth-order prerequisites before any geometry construction.
curved_transverse_whole_reference_keeps_profile_identity|Does the complete-piece certificate identify which proof actually ran?|Require the selected private proof switch to reach construction and remain in the result together with the exact piece and motion identity.
curved_transverse_real_cover_forwards_each_chart_and_keeps_motion|Is every real chart bound to the declared curved correction profile?|Build the original pieceÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢s fresh complete cover and check every chart receives the explicit fourth-order curved/transverse settings without changing its motion direction.
""")

_add('packaging/test_opaque_tangent_transverse', 'Private reference-proof cost controls; retain every certified crossing and tiny term, with independent closed integral checks in the same suite.', """
curved_quadrature_lines_keep_crossings_without_optional_newton_work|Can we avoid repeated root calculations without losing a boundary contribution?|Require the original certified quadrature-line enclosure and all four jump orders to remain while optional Newton contraction is not invoked. Keep a tiny signed motion contribution.
transverse_empty_jump_list_reuses_complete_smooth_proof_without_field_work|Does the proof skip calculations only when there is no jump to refine?|Provide a complete smooth bound with a tiny nonzero derivative and an empty jump list; forbid extra field evaluations and require the same bound object back.
""")

_add('packaging/test_opaque_motion_extension', 'Fresh independent inside-horizon reference using its existing certified root map; this is a separate profile from outside-horizon angular inversion.', """
remaining_inside_curved_fresh_reference|Does each remaining inside-horizon piece have a complete motion integral?|Verify its inside-region identity, rebuild its own field and moving horizon, retain all ordinary and boundary contributions, and require the unchanged complete uncertainty and cell/depth/time limits. A partial cover stays unresolved.
""")


_add('packaging/test_opaque_inside_angle', 'Independent inside-horizon map/coverage and shared-shading controls. Complete-piece uncertainty, installed rendering, world motion and full-frame gates remain separate.', """
constant_edge_cover_preserves_area_and_tiny_motion|Does following the shading boundaries preserve all area and very small motion?|Integrate the complete constant-edge band cover against exact area and first angular-moment formulas, including signed 2^-60 motion. No region or contribution is omitted.
constant_edge_cover_refuses_missing_proofs|Can an unproved map silently enter the checker?|Reject a varying transverse edge, a nonpositive radial slope, a moving zero horizon, or a level outside the certified positive edge.
inside_prepared_angle_retains_actual_horizon_and_motion|Does the reversed inside coordinate still represent the original surfaces?|Compare all mixed field coefficients to the independently assembled original inside map for both pieces and all three motions, keep the exact zero horizon, and verify the physical-origin edge and analytic derivative.
inside_actual_angular_cover_has_complete_measure_and_matched_motion|Do the new bands cover both actual inside pieces completely?|Verify strict radial monotonicity, constant transverse edge, every chart's independent angle/motion and positive physical measure, with the full balanced cover and no missing wedge.
inside_profile_refuses_noninside_geometry|Can an outside surface accidentally use this inside proof?|Require explicit refusal for front-facing and outside-horizon original pieces; keep the old outside-only constructor guards.
inside_shared_shading_callbacks_match_independent_vector_radiance|Does the shared checker evaluate the correct material after changing coordinates?|Compare RGB and its first motion against an independently assembled normal/view vector Principled formula on horizon, interior and origin-edge charts of both pieces.
inside_complete_reference_keeps_identity_and_terminal_failure|Can a failed inside integration produce a usable certificate?|Verify exact piece/face/motion/anchor identity in a completed result and retain an explicit unresolved failure instead of returning a partial sum.
inside_unsquared_residual_retains_independent_mixed_field|Does the cheaper inverse preserve the correct inside geometry and tiny derivatives?|Compare the unsquared algebraic residual's spatial and first-motion coefficients to an independent vector norm/dot expression, retaining the signed negative physical width and a 2^-60 target motion.
""")
_add('packaging/test_opaque_motion_extension', 'Fresh independent complete inside-horizon material reference. No installed, whole-pixel or world-geometry admission; retain every contribution and the unchanged limits.', """
inside_angular_piece_fresh_reference|Does the complete inside piece fit the accuracy and checking budget?|Build a fresh constant-edge angular cover and integrate all smooth, ordinary and boundary-motion terms under one construction-inclusive clock. Verify the final RGB reference radius; any deadline, cell, depth or uncertainty failure remains unresolved.
""")

_add('packaging/test_opaque_inside_angle', 'Exact reversed physical-coordinate inverse and retained first motion. These controls do not admit a complete material integral or installed frame.', """
inside_physical_inverse_matches_independent_field_and_known_motion|Does the faster physical-coordinate inverse preserve the original geometry and tiny motion?|Substitute the reversed radius into an independent vector dot/norm expression, verify the complete residual, recover a known radius and its spatial/mixed coefficients, and retain a signed 2^-60 motion. Certify the positive whole-bracket slope.
""")

_add('packaging/test_opaque_motion_extension', 'Stronger selected front reference targets under the existing cell, depth, worker and memory limits. Default target and production rendering remain unchanged.', """
stronger_front_reference_tolerance_rejects_looser_or_invalid_values|Can a selected proof silently relax accuracy?|Reject looser targets, zero, negative, nonfinite or malformed values and non-string inputs. An opt-in can request only a stronger finite positive target than the existing default.
stronger_front_reference_tolerance_is_forwarded_without_changing_limits|Does the stronger target reach the checker without increasing its budget?|Verify exact target forwarding to the complete-piece integrator and certificate, with the same 256-cell and depth-10 caps. These controls admit target selection, not a numerical reference or renderer result.
""")

_add('packaging/test_opaque_face_certificate', 'Complete selected-pixel references retain every independently admitted piece for the same motion. Passing separate faces never bypasses whole-pixel uncertainty or byte/input checks.', """
pixel_sum_preserves_every_signed_tiny_piece|Does the full reference retain all thirteen tiny signed contributions?|Compose every piece and face outward, then compare the RGB value and derivative endpoints to exact rational sums. No small contribution is discarded.
pixel_sum_refuses_incomplete_mismatched_or_invalid_evidence|Can a missing or mismatched piece be accepted as a complete pixel?|Reject missing, duplicated, wrong-motion, wrong-face or failed evidence and malformed/nonfinite or too-wide intervals before admitting the full pixel.
pixel_sum_refuses_combined_uncertainty_even_when_every_face_passes|Can passing faces conceal excessive full-pixel uncertainty?|Create a control where every complete face passes individually but the total exceeds the unchanged1e-12radius. The full pixel must still be refused.
pixel_bundle_checks_input_artifact_bytes_and_recomputed_sum|Is the full-pixel reference tied to exact inputs and artifact bytes?|Verify the bundle and every certificate digest, exact input identity, all thirteen pieces and a recomputed outward sum. Refuse stale sums, tampering, missing or relabeled evidence.
""")
_add('packaging/test_opaque_streamed_motion', 'Complete-pixel diagnostic aggregation with every active face and signed tiny value retained. This does not change native rendering or admit a whole frame.', """
complete_pixel_face_sum_keeps_signed_tiny_contributions|Does combining faces preserve tiny values despite large cancellation?|Use exact binary tiny values together with large opposing terms. Verify every RGB value/derivative and bounded sample-VJP count, including reversed face order.
complete_pixel_face_sum_refuses_missing_owners_or_invalid_values|Can a complete pixel silently omit an active rendered face?|Reject missing, duplicate or unexpected face identities, duplicate expectations, malformed channels and nonfinite values or derivatives.
""")
_add('python/test_opaque_piece_motion_acceptance', 'Installed-wheel comparison for the complete selected original pixel and one declared packed-input motion. No whole-frame/world-geometry/camera/recomputed-normal admission.', """
original_motion_meets_independent_complete_pixel|Does the complete original pixel have the independently verified colour and motion gradient?|Check the original input and all thirteen reference artifacts, require every active native owner to match the six certified faces, evaluate all samples with the installed shader/VJP, and compare all RGB values/derivatives to both independently bounded endpoints. Retain all accuracy and worker/memory/sample limits; qualify CPU/CUDA bytes separately.
""")

_add('packaging/test_opaque_motion_extension', 'Bounded diagnostic on three saved regions of a failed stronger normal reference. Component bounds and local timing do not qualify a complete piece or pixel.', """
failed_normal_saved_refinement_directions|Which split reduces the remaining normal boundary uncertainty?|Check the frozen failed trial and its three largest retained regions. Recompute parent contribution bounds and both exact radial/transverse child covers under one bounded clock. Preserve all signed terms and report the summed child widths. Complete smooth/error and whole-piece timing still require separate qualification before another full trial.
""")


_add('packaging/test_opaque_motion_extension', 'Independent local refinement observations and front-only work-order controls. Hints and local gains never admit a complete piece or change any bound or resource limit.', """
remaining_front_saved_refinement_directions|Which direction should refine the remaining front-piece proofs?|Check three retained regions each for pieces 2, 3 and 10, recompute parent and both exact child covers under one clock, and report complete correction widths. Historical evidence chooses geometry only; a new complete proof remains necessary.
front_fixed_refinement_axis_refuses_invalid_or_unqualified_selection|Can an ambiguous hint alter the proof profile?|Reject invalid axis types and unqualified profiles before geometry construction.
front_fixed_refinement_axis_is_geometry_only|Does choosing a split direction change a calculated result?|Supply controlled complete bounds, require their endpoints to remain unchanged and verify that only the refinement hint changes. Keep the original acceptance, cell and depth limits.
front_fixed_refinement_axis_refuses_a_curved_chart|Can a front-only hint silently change the curved proof?|Reject the unqualified curved use explicitly; curved motion retains its separate mapping and integration obligations.
""")
_add('packaging/test_opaque_polynomial_inputs', 'Exact rational cancellation followed by full outward spatial/first-parameter jets. Closed Taylor formulas qualify the algebra; selected original-input comparisons check every coefficient. Complete-piece and installed acceptance are separate.', """
rotating_normal_has_closed_full_taylor_and_tiny_motion|Can exact rotational norm cancellation lose a tiny gradient?|Compare all coefficients through order 10 to independent binomial Taylor formulas for a rotating normal, including a negative motion of 1e-30. Intrinsic norm cancellation must be algebraic, while all remaining mixed coefficients stay present.
moving_view_has_closed_full_taylor|Does the same algebra retain moving-view derivatives?|Use an independent closed normalized-view Taylor expansion and verify both spatial and first-parameter coefficients, including very small nonzero motion.
coordinate_motion_retains_all_mixed_terms_after_norm_cancellation|Does cancelling intrinsic norm motion accidentally cancel coordinate motion?|Give the radial coordinate a separate nonzero motion and check every mixed coefficient through order 10 against an independent closed chain-rule formula.
polynomial_inputs_refuse_invalid_norms_or_jets|Can the model evaluate an undefined normalized direction?|Reject zero normal/view norms, mismatched jet orders and malformed field shapes.
original_front_inputs_keep_all_coefficients_and_jacobian|Does the repair still describe the original mesh and all declared motions?|Compare every old and exact-polynomial input coefficient for normal, view and attached-screen motion at orders 0, 1, 4 and 10. Require the physical Jacobian endpoints to remain exactly unchanged.
polynomial_reference_rejects_invalid_or_unqualified_profile|Can the checker activate this algebra without its qualified prerequisites?|Reject ambiguous switches and missing conditioned, transverse and material-input proof settings before construction.
polynomial_reference_uses_full_model_and_original_jacobian|Does the selected shared checker really use the qualified model?|Forbid the old evaluator, invoke the selected full model through the real checker callbacks, and verify motion identity, unchanged acceptance and work limits.
polynomial_reference_refuses_unqualified_curved_mapping|Can a front-only repair substitute an unproved curved-horizon model?|Reject curved activation explicitly. Moving horizons retain their separate certified maps and Jacobians.
saved_front_polynomial_input_cost_and_bounds|Does the algebra reduce actual checking cost and uncertainty?|Check the exact retained evidence and alternate old/new timing on nine saved regions. Report both complete correction components and elapsed time; local gains do not qualify a whole piece or frame.
""")


_add('packaging/test_opaque_curved_tolerance', 'Explicit stronger private accuracy targets for both curved reference families. Defaults and the full-pixel acceptance criterion remain unchanged; no work, clock or memory reset is allowed.', """
curved_reference_refuses_invalid_or_looser_proof_targets|Can a private proof target weaken the original accuracy requirement?|Reject missing, non-string, nonfinite, nonpositive or looser targets before geometry construction, for both outside and inside horizon families.
curved_reference_forwards_stronger_target_without_budget_reset|Can a tighter target secretly grant a larger budget?|Require the selected target to reach the shared complete-cover integrator and result while retaining 256 cells, depth 10 and the single construction-inclusive clock. Original RGB/full-pixel acceptance is unchanged.
""")


_add('packaging/test_opaque_polynomial_inputs', 'Independent closed light/half-vector Taylor controls. Full coordinate and first-motion chains are retained; complete-piece accuracy and runtime require their own fresh proof.', """
light_only_keeps_closed_coordinate_chain_without_unused_view_work|Can a faster material query lose a tiny moving-coordinate contribution?|Compare every spatial/mixed light coefficient through order 10 to independent binomial formulas with nonzero coordinate motion. Forbid unused view and normal-view-dot evaluation only when the caller consumes light alone; the geometry cover still owns valid view norms.
light_highlight_rejects_invalid_inputs|Can an invalid light query enter the checker?|Reject mismatched full jets, ambiguous material-only flags and a nonpositive normal norm.
physical_light_highlight_keeps_original_full_coefficients|Does the polynomial light model retain the original curved surface fields?|Compare light and highlight coefficients to independently assembled vector fields at orders 0, 1, 4 and 10 for normal, view and attached-screen motion. Retain the original curved map and every mixed term.
""")
_add('packaging/test_opaque_polynomial_light_binding', 'Explicit private curved/inside checker integration; the certified inverse, physical measure, horizon fallback and all existing limits remain. Local timings never admit a complete piece, installed pixel or milestone.', """
polynomial_light_refuses_unqualified_outside_before_geometry|Can the new algebra activate on an unqualified map?|Reject ambiguous switches and missing conditioned fourth-order physical-angle material/transverse prerequisites before geometry construction.
polynomial_light_complete_and_failed_identity|Can a partial or failed proof appear as complete evidence?|Verify exact profile and motion identity in complete and unresolved results, including tighter targets under the original cell/depth and construction-clock limits.
polynomial_light_physical_callbacks_keep_angle_measure_and_mixed_motion|Does the real checker retain the same angular target, area and moving horizon?|Build fresh outside and inside covers for all three motions; require exact angular and physical-Jacobian endpoints and overlapping full light/shading coefficients. Smooth radiance must consume the full highlight; material-only queries omit only unused view work.
saved_curved_polynomial_light_cost_and_complete_bound_components|Does the repair improve checking time while retaining every bound component?|Pin three admitted saved regions, rebuild old/new covers and alternate timing. Recompute ordinary residual, combined tangent and both smooth remainder orders using each chart's actual quadrature rule. Report local cost and bounds; a fresh complete trial still decides admission.
""")


_add('packaging/test_opaque_polynomial_light_binding', 'Exact horizon-query consumption controls. Skip only the highlight that the material correction does not consume; full smooth radiance still computes it. No changed accuracy or resource requirement.', """
horizon_material_query_forwards_only_consumed_fields|Can the horizon checker avoid unused highlight work without changing its proof?|Forward the explicit material-only flag through real outside/inside horizon callbacks for all three motions at spatial orders 1 and 4. Require every consumed angle, light and Jacobian endpoint to match the full evaluator exactly; require smooth radiance to retain its full highlight.
""")

_add('packaging/test_opaque_checker_cost', 'Bounded checker algebra and saved-region diagnostics; local timings are not complete-piece acceptance.', """
saved_piece8_cost|Where does independent checking spend its time?|Pin three saved complete regions, reconstruct the real cover and measure all proof components including corrections and transport. Retain bounds and source identities; a separate full proof decides acceptance.
chart_cache_keeps_unclipped_values_and_fresh_local_bounds|Can a reused chart value carry a stale local bound?|Compare every endpoint with the frozen original arithmetic across changed local regions, inside/outside pieces and all motions; require private returned values and the existing cache cap.
cancelled_physical_slope_keeps_closed_mixed_coefficients|Does a polynomial slope retain tiny mixed motion?|Compare every coefficient to independent binomial formulas for both horizon orientations and signed tiny motion. This alternative is qualified algebra but remains unselected after its slower diagnostic.
cancelled_physical_slope_removes_interval_self_dependency|Can exact cancellation tighten a slope without dropping terms?|Compare the polynomial formula to a closed analytic interval range. No numerical contribution is thresholded away.
shared_inverse_slope_keeps_closed_root_measure_and_mixed_motion|Can sharing the inverse also preserve its full moving area factor?|Compare root, physical slope and reciprocal measure to independent closed formulas through order10 on both coordinate axes and both horizon orientations, including signed1e-30 motion.
shared_inverse_slope_rejects_unqualified_query|Can the optimization bypass profile guards?|Require an explicit compiled profile and reject ambiguous switches before shared-slope evaluation.
""")
_add('python/test_opaque_public_motion', 'Actual public original-scene motion named in each result against a complete source-bound independent pixel. A passing direction does not admit unrelated deformations or full frames.', """
public_original_motion_against_complete_reference|Do the declared public scene motions produce the independently expected image change?|Load all13 same-motion references, preserve the original full material and pixel, and compare every RGB value and derivative to both outward endpoints. Check device bytes separately. The normal scalar uses an explicit fixed-order caller reduction; its earlier ordinary-broadcast mismatch remains historical.
""")

_add('packaging/test_opaque_projected_motion', 'Independent geometry and proof-checker controls. Each result has a bounded scope; a local proof or timing does not qualify a complete original pixel.', """
moving_projected_basis_has_exact_vertex_identities|Does moving projection preserve the triangle coordinates?|Compare exact rational moving barycentric numerators at the triangle vertices.
moving_projected_fields_keep_closed_perspective_and_horizon_motion|Does the independent field retain perspective and horizon motion together?|Use closed polynomial identities for moving normal/view numerators and the horizon equation.
moving_projected_fields_refuse_invalid_geometry|Can ambiguous geometry become a reference?|Reject degenerate triangles and invalid perspective weights before integration.
aligned_world_motion_has_closed_projection_and_view_chain|Does actual world motion change both projection and view?|Check exact aligned-camera translation and shear chains, including shared vertices and fixed depth.
affine_shear_recomputed_area_normals_follow_exact_cofactor|How should normals change under shear?|Compare transformed triangle area vectors with the independent cofactor identity.
original_world_prepare_keeps_base_fields_and_complete_motion|Does the moving reference describe the original mesh?|Require unchanged base fields and the complete physical first-motion chain for the pinned source profile.
world_reference_forwards_explicit_profile_without_relabeling|Can a packed-input check be relabelled as world motion?|Require the explicit physical motion profile to reach the correct reference family.
original_world_fresh_reference|Is this complete physical-motion piece independently accurate?|Recompute every region under the original time, memory, depth and cell limits; an unfinished piece remains unresolved.
original_world_physical_map_keeps_implicit_angle_and_complete_measure|Does the curved coordinate transformation retain its moving area?|Check the implicit angular equation and complete Jacobian including physical motion.
quadratic_first_motion_product_does_not_construct_discarded_second_motion|Can first-order polynomial work skip irrelevant higher motion?|Compare consumed constant and first coefficients exactly without constructing discarded second-order products.
quadratic_first_motion_product_retains_complexity_guard_for_consumed_terms|Does this optimization bypass polynomial limits?|Require the existing rational and degree guards on every coefficient actually consumed.
normalized_quadratic_root_preserves_closed_large_rational_scale|Can root normalization change a tiny contribution?|Check closed roots with large exact rational scales, retaining nonzero terms.
normalized_quadratic_root_refuses_ambiguous_or_nonpositive_scale|When is root normalization invalid?|Reject unproved or nonpositive scaling before forming the root.
world_shear_normalization_precedes_perspective_interpolation|Are each vertex's recomputed normals treated correctly?|Differentiate per-vertex normalization before interpolation and compare the full chain.
world_angular_correction_direction_reaches_every_chart|Does the selected integration direction reach the whole cover?|Capture all chart corrections and require the explicit axis selection on each one.
angular_correction_direction_refuses_invalid_selection|Can an invalid axis silently select a method?|Reject unsupported direction values before proof work.
local_crossing_range_excludes_only_provably_absent_knot|Can a local range omit a possible shading crossing?|Exclude only knots proven outside the full enclosure; uncertain crossings remain included.
local_crossing_range_requires_explicit_boolean|Is local crossing selection explicit?|Reject ambiguous switches rather than enabling a new proof path implicitly.
local_crossing_proof_retains_closed_piecewise_integrals|Does local support preserve the integral across knots?|Compare against independent piecewise closed integrals including signed contributions.
sixth_line_proof_keeps_closed_log_integrals|Can higher-order line bounds miss logarithmic terms?|Enclose closed logarithmic integrals with signed and tiny nonzero coefficients.
mixed_fourth_sixth_rule_keeps_all_closed_jumps|Does combining different axis orders retain every jump?|Check closed piecewise functions with the complete jump terms for each axis rule.
sixth_line_proof_rejects_unqualified_profiles|Can the higher-order rule be selected outside its proof?|Reject unsupported order and profile combinations.
identical_rgb_transport_reuses_only_equal_endpoint_proofs|Can RGB channels share transport calculations safely?|Reuse only exactly identical endpoint inputs and preserve distinct channels and overrides.
front_line_order_reaches_qualified_correction|Does the reference really use the selected line order?|Capture the correction constructed by the full front-piece reference.
front_line_order_refuses_invalid_selection|Can invalid line orders enter a trial?|Require explicit supported orders before numerical work.
cancelled_polynomial_motion_preserves_complete_coordinate_chain|Does symbolic cancellation keep mixed geometry derivatives?|Compare the full closed coordinate chain including tiny signed motion terms.
cancelled_polynomial_motion_requires_explicit_boolean|Is the alternative motion formula opt-in?|Reject ambiguous selection values; qualification does not imply production adoption.
crossing_support_keeps_every_possible_transverse_root|Can a support contraction lose a crossing?|Require the contracted bracket to contain every root supported by the original domain.
crossing_support_preserves_closed_log_integrals|Does contraction retain both integral and derivative?|Enclose independent closed logarithmic examples with signed motion.
crossing_support_requires_explicit_boolean|Can contraction turn on accidentally?|Reject nonboolean profile selections.
front_crossing_support_reaches_qualified_correction|Does the full reference use the intended contraction?|Capture the actual constructed correction and its explicit switch.
front_crossing_support_refuses_invalid_selection|Are unsupported contraction profiles refused?|Reject invalid inputs before construction.
front_material_cache_preserves_fresh_region_bounds|Can shared raw fields retain stale regional clamps?|Compare adjacent regions with the frozen calculation while applying each region's newly proved bounds.
partitioned_crossing_keeps_closed_two_axis_log_integral|Does the tensor proof enclose a known two-dimensional integral?|Check independent closed log integrals across both integration axes, spatial orientations and tiny signed terms. These controls exposed the earlier insufficient preliminary bound.
root_partition_requires_qualified_fixed_count|Can extra root partitions expand work without a rule?|Allow only the qualified fixed partition counts and valid proof profiles.
requalify_saved_tensor_certificate|Does a repaired checker independently replace an old certificate?|Use only the old geometry and explicit profile; recompute all numerical bounds with the repaired identity and original limits.
polynomial_preparation_omits_only_unconsumed_scalar_work|Does preparation avoid duplicate work without changing values?|Compare all consumed endpoints with frozen arithmetic over point and interval inputs and several derivative orders.
polynomial_prepare_rebuilds_on_context_changes_without_evaluating|Can prepared coefficients survive a precision change incorrectly?|Change arithmetic context, require reconstruction and verify no discarded scalar evaluation occurs.
weighted_kernel_reuse_preserves_all_frozen_endpoints|Does sharing kernel geometry preserve the exact bound?|Compare frozen endpoints for signed and uncertain weights, including tiny nonzero supports.
weighted_kernel_cache_respects_context_and_capacity|Can kernel reuse cross a precision boundary or grow without limit?|Require context-sensitive keys and the fixed128-entry bound.
front_root_partition_reaches_qualified_correction|Does the full reference apply the selected root partition?|Capture the constructed correction and check the explicit fixed count.
front_root_partition_refuses_unqualified_profile|Can root partitioning bypass its assumptions?|Reject unsupported profiles before constructing the correction.
front_full_samples_share_only_identical_raw_material_fields|Can smooth and residual calculations reuse the same sample fields?|Require exact frozen endpoints and remove only repeated raw material evaluations; each region still applies its own bounds.
material_seed_freezes_fields_and_rejects_conflicting_values|Can a caller corrupt shared sample data?|Store immutable endpoints, return private copies and refuse a conflicting value under an existing key.
""")
_add('packaging/test_opaque_checker_cost', 'Saved-region diagnostics and proof-path audits. These are scoped investigations, never complete-piece acceptance.', """
world_saved_piece8_cost|Where is physical curved-piece checking time spent?|Measure three pinned regions while retaining every original correction, remainder and transport contribution.
world_front_saved_corrections|Which front-piece correction terms dominate the bound?|Recompute three pinned regions or exact sibling parents with explicit proof options; retain each axis contribution separately.
saved_tensor_proof_path_audit|Which historical certificates used the defective preliminary bound?|Reconstruct each saved leaf and classify its actual proof selection. This audit identifies requalification work without claiming new numerical accuracy.
world_front_merge_domain_preflight|Can two saved siblings share a coarser proof region?|Check the merged parent's angular domain against the unchanged segment cap. Rejected parents retain their original children.
""")

_add('packaging/test_opaque_face_certificate', 'Complete physical motion proofs must bind their exact input and source profile, retain every piece and meet the unchanged combined uncertainty gate.', """
physical_pixel_sum_binds_input_and_world_source|Can physical-motion certificates be composed without losing their identity?|Bind all thirteen pieces to the expected input and world/camera profile digests; preserve signed tiny contributions.
physical_pixel_sum_refuses_unbound_or_incomplete_evidence|Can missing, mixed or overly uncertain pieces pass as a full pixel?|Reject missing identities, invalid hashes, packed-motion substitutions, duplicates and a sum that exceeds the pixel bound.
physical_pixel_bundle_requires_expected_source_identity|Does a stored bundle match the source the caller intended?|Recheck every artifact digest and recompute the sum, rejecting changed input or world-profile identities even when a file has a new valid digest.
""")
_add('packaging/test_opaque_worker_identity', 'Worker provenance and thread enforcement are required before public installed-wheel tests execute.', """
public_worker_checks_wheel_and_thread_limit_before_tests|Can a public test run against unverified installed files?|Exercise the real worker entry point with controlled dependencies; require one-thread settings and wheel-member verification before the test runner, and prevent execution when identity fails.
""")

_add('python/test_native_opaque_transport', 'Native weighted-integral transport only. Exact polynomial controls and device bytes do not establish arbitrary shader accuracy; sample capacity and first-derivative limits remain.', '\nmaterial_transport_fixed_pixel_has_closed_polynomial_shape_derivatives|Does light transported through a fixed pixel have the right motion derivative?|Move a triangle carrying known polynomial colours and compare image integrals and boundary-flux derivatives with exact formulas.\nmaterial_transport_visible_silhouette_has_closed_area_derivative|Does a moving silhouette change covered area by the correct amount?|Compare a clipped moving triangle with its exact visible area and area derivative.\nmaterial_transport_preserves_first_only_and_capacity_refusal|Can transport bypass derivative or geometry limits?|Require explicit rejection of second derivatives and too many candidate faces.\nmaterial_transport_capacity_retains_area_and_fallback|Does changing the sample budget retain covered area and its gradient?|Check exact areas and vertex derivatives across the transport capacity boundary and the raw-rule fallback.\nmaterial_transport_cpu_cuda_sample_and_first_vjp_bytes|Do CPU and GPU execute the same native transport arithmetic?|Compare every owner, sample and vertex/cut first-derivative byte with identical supplied upstream gradients in both supported precisions.\nmaterial_transport_linear_cut_retains_one_sided_interior_flux|Does a moving colour jump retain its missing boundary contribution?|Integrate an exact moving step function; require its area derivative and the same interior samples as the raw quadrature rule.\n')
_add('python/test_opaque_piece_motion_acceptance', 'Research diagnostics retain the original independent intervals, every contribution and existing caps. A successful diagnostic runner is not renderer acceptance when an endpoint comparison fails.', '\ncomplete_physical_pixel_diagnostic|Which physical world-motion contributions explain the renderer error?|Decompose the complete original pixel into its visible faces and compare each packed-input result with the source-bound independent pixel.\nphysical_flat_reallocation_diagnostic|Would changing interior sample placement repair the gradient?|Reallocate only the declared all-flat faces, keep curved faces unchanged and report independent endpoint errors separately from test execution.\nphysical_symmetric_triangle_diagnostic|Would a symmetric triangle rule fix the directional sampling error?|Use the pinned positive degree-fourteen rule on declared flat faces and retain per-face endpoint errors; this experiment did not establish production acceptance.\nphysical_flat_transport_diagnostic|Does accounting for material flow and boundary flux repair geometry motion?|Use exact rational fragment geometry for a research transport calculation and compare the complete pixel with its independent reference.\ntransport_rebalanced_pixel_diagnostic|Can interior and edge samples be balanced without increasing the pixel cap?|Compare explicitly bounded allocations on both normal and world motion; preserve failed alternatives and require separate native qualification.\n')
_add('python/test_opaque_public_motion', 'Actual source identity and declared motion family only. Source extraction is not an image or gradient acceptance result.', '\nextract_original_world_motion_profile|Does the independent motion model describe the real original scene?|Verify the actual camera pose, selected world vertices, projected corners, normals and view vectors against the frozen input before saving the profile.\n')
_add('packaging/test_opaque_camera_reversal', 'An exact reversal for the pinned aligned camera, constant material and directional illumination only; it is not a proof for arbitrary camera motion.', '\ncamera_reversal_preserves_all_values_and_signed_tiny_derivatives|Can the world-translation reference rigorously check opposite camera translation?|Verify the source-bound algebraic reversal, retain values and negate both derivative endpoints, including tiny nonzero terms.\ncamera_reversal_refuses_unproved_or_mismatched_source|Could a reference be relabeled for an unrelated camera or scene?|Mutate source identities, motions and required assumptions and require rejection before deriving a camera reference.\n')
_add('packaging/test_opaque_flat_resample', 'Closed diagnostic resampling controls only; quadrature moment agreement does not establish nonlinear material or complete renderer accuracy.', '\nflat_reallocation_from_existing_five_by_eight_keeps_moments|Does a changed sample layout still describe the same triangle?|Reconstruct the existing five-by-eight affine fan and compare exact polynomial moments under the bounded alternate layouts.\n')
_add('packaging/test_opaque_flat_transport', 'Independent polynomial transport identities for affine source flow and declared rules only; nonlinear shading needs its separate certified comparisons.', '\nflat_transport_matches_closed_material_and_boundary_derivatives|Are both material flow and moving-boundary terms present?|Compare fixed clips, moving clips and affine expansion against exact polynomial integrals and derivatives for each bounded rule.\nflat_transport_refuses_degenerate_over_capacity_and_nonfinite|Can invalid geometry or excessive work produce a plausible transport result?|Require rejection of degenerate source triangles and fragment counts beyond the declared diagnostic capacity.\n')
_add('packaging/test_opaque_symmetric_triangle', 'Source-pinned symmetric-rule research only. Closed moments do not promote the failed original-gradient experiment or replace independent nonlinear bounds.', '\nsymmetric_triangle_has_positive_interior_rule_and_all_degree14_moments|Does the imported triangle table implement its stated rule?|Check the exact table identity, positive interior samples and all one hundred twenty factorial polynomial moments through degree fourteen.\nsymmetric_resampling_keeps_moving_triangle_moments|Does mapping the symmetric rule preserve moving triangle geometry?|Compare closed values and first motions after affine reconstruction and require refusal of unsupported input layouts.\n')

_add('packaging/test_opaque_polynomial_inputs', 'Context-owned polynomial reuse preserves existing outward arithmetic; local timing improvements do not admit a complete reference or change its deadline.', """
polynomial_row_reuse_keeps_frozen_endpoints_and_private_values|Can repeated row calculations be shared without changing any bound?|Compare all ordered coefficients with the frozen Horner calculation through order ten, retain tiny signed motion, and mutate returned results to check cache isolation.
polynomial_row_reuse_bounds_storage_and_rebuilds_after_context_change|Could reuse grow without limit or survive an arithmetic change incorrectly?|Exercise more than sixty-four inputs, require the fixed cache limit and force a fresh evaluation after changing the arithmetic context.
polynomial_horner_omits_only_leading_exact_zero_products|Can setup overhead be removed without discarding a small real contribution?|Count initial zero products while retaining the analytically negative nonzero tiny derivative; endpoint controls separately compare the unchanged calculation.
""")

_add('packaging/test_opaque_projected_motion', 'These checker controls qualify bounded mathematical identities and explicit profile selection. A complete original-piece trial and installed renderer comparison are still required.', """
curved_sixth_lines_keep_closed_log_integrals|Does a stronger curved line proof preserve every logarithmic contribution?|Check independent closed values and derivatives across knots in both directions and orientations.
curved_sixth_lines_keep_nonlinear_change_of_variable|Does the curved proof include the complete coordinate Jacobian?|Transform a quadratic coordinate with its exact measure and compare independent closed integrals, including tiny negative motion.
curved_partitioned_roots_keep_closed_nonlinear_integral|Can tighter crossing brackets lose part of a curved boundary?|Check a nonlinear two-dimensional change of variables against closed value and gradient integrals, retaining every selected two- or four-part root-proof domain.
explicit_pure_flux_reuse_keeps_every_endpoint_and_tiny_term|Can identical RGB flux proofs be calculated once?|Require an explicit pure callback, compare every result endpoint with separate calculations, and retain distinct channels and tiny signed terms.
pure_flux_reuse_refuses_implicit_contract|Can a stateful callback silently be treated as pure?|Reject nonboolean or missing-callback purity claims; the default still invokes all channels separately.
curved_root_profile_reaches_full_checker|Does the complete checker use the qualified curved proof?|Capture both cover construction and the complete-piece wrapper and verify explicit support and partition settings.
curved_root_profile_rejects_unqualified_selection|Can unsupported proof settings enter a full trial?|Reject invalid partition counts and incomplete curved-profile combinations before numerical work.
""")
_add('packaging/test_opaque_tensor_smooth', 'Local derivative bounds retain the original quadrature nodes, complete domain and every correction. These proof checks do not by themselves qualify a mesh piece.', """
radial_quarter_peano_masses_enclose_independent_antiderivative|How do we know the smaller proof intervals cover the full integration error?|Check all four radial kernel masses against an independent integrated polynomial formula on both arithmetic backends.
biaxis_local_tensor_preserves_closed_integrals_and_all_contributions|Can tighter smooth bounds discard a small value or gradient?|Check exact polynomial and rational integrals, signed tiny motions, the same twenty-four samples and both separate correction terms.
""")

_add('packaging/test_opaque_checker_cost', 'Cost diagnostics guide new work layouts only; they never admit a complete piece or change acceptance.', """
saved_shear_cover_natural_range_cost|Which small regions have a cheaper complete residual proof?|Measure signed natural-range intervals across the source-bound saved cover, keeping smooth radiance and motion flux as separate unfinished obligations.
""")
_add('packaging/test_opaque_tangent_peano', 'Proof selection changes calculation cost; the complete summed error still owns acceptance.', """
conditioned_range_selection_retains_complete_small_integral|Does the cheap small-residual path keep the contribution?|Compare complete signed intervals with closed linear integrals, including a tiny nonzero coefficient; no term is replaced by zero.
""")
_add('packaging/test_opaque_tangent_transverse', 'Optional root refinement never removes the original complete crossing enclosure.', """
partitioned_transverse_skips_only_already_enclosed_small_jump|Can optional tightening be skipped safely?|Require the unchanged full signed tiny-jump interval while making any unnecessary partition calculation fail the test.
""")
_add('packaging/test_opaque_polynomial_inputs', 'Reuse stores only bounded immutable coefficients keyed to the exact domain and arithmetic context; whole-piece runtime remains a separate test.', """
angle_differential_reuses_only_identical_scalar_domains|Can revisiting an interval reuse shading coefficients without changing the derivative?|Check a closed scalar derivative and tiny signed motion, mutate a returned value, exceed the cache capacity and change precision to verify isolation and invalidation.
""")

_add('packaging/test_opaque_checker_cost', 'Independent checker controls; a local speed or identity test does not admit a complete mesh piece.', """
curved_full_samples_seed_identical_raw_material_fields|Can smooth samples safely supply repeated material fields?|Compare every raw coefficient against a frozen implementation across four chart types, preserving private copies and every tiny term.
scalar_implicit_inverse_preserves_closed_motion_without_series|Can point evaluation avoid building a spatial power series?|Compare radius and radial derivatives with independent closed formulas, including intrinsic motion, uncertain inputs and tiny signed motion.
""")
_add('packaging/test_opaque_tangent_transverse', 'Optional numerical tightening never removes the full certified root interval.', """
crossing_contraction_stops_after_sufficient_enclosed_reduction|Can root tightening stop before an arbitrary number of iterations?|Retain the entire nonzero bracket and independently check nonlinear roots in both axes and orientations; the whole-piece error gate remains unchanged.
""")
_add('packaging/test_opaque_polynomial_inputs', 'Arithmetic reuse preserves separately directed operations and exact frozen endpoints.', """
jet_addition_shares_context_without_changing_frozen_endpoints|Can coefficient addition share rounding setup?|Check orders zero, six and ten at three precisions against the frozen implementation, including signed tiny values and returned-object isolation.
""")
_add('packaging/test_opaque_tensor_smooth', 'Stronger rules must prove their full error and pay their sample cost within the existing work budget. These controls alone do not qualify a complete mesh or final showcase.', """
gauss12_positive_rule_independent_moments_and_kernels|Why trust the stronger transverse integration rule?|Isolate every root with exact rational brackets, check positive weights and all24polynomial moments, then verify each quarter kernel mass against an independent antiderivative on both arithmetic backends.
tensor36_complete_integrals_corrections_and_work_charge|Does stronger sampling retain all corrections and honest resource charges?|Check closed full integrals and tiny signed correction terms; require36samples to consume two work units even for failed attempts, including mixed-rule covers.
tensor36_saved_shear_regions_combined_bounds|Does stronger integration help on the actual difficult mesh regions?|Compare complete enclosures on three identical saved domains with all smooth, residual and moving-boundary terms; report both error and time without claiming whole-piece acceptance.
tensor36_chart_selection_is_explicit_and_bound|Can a full trial silently choose different rules?|Require explicit unique chart ids, verify selected and unselected charts, reject invalid choices and bind the selection through the full-piece wrapper.
tensor48_radial_moments_full_integrals_and_work|Can the radial error be tightened within the same work allowance?|Check independent Gauss4moments, closed integrals and tiny signed contributions, with48samples charged as two work units and refusal when only one unit remains.
""")

_add('packaging/test_opaque_polynomial_inputs', 'The scalar Horner repair preserves each separately directed product and addition; it does not fuse or reorder arithmetic.', """
prepared_scalar_horner_shares_context_and_keeps_frozen_endpoints|Can scalar polynomial evaluation avoid repeated setup and temporary products?|Require every endpoint to equal the frozen implementation across three precisions, all sign patterns, tiny terms and repeated calls; returned intervals remain private.
""")

_add('packaging/test_opaque_batch_cursor','Internal C11 continuation accounting only. Production shading, installed wheel, frame timing and accuracy require separate qualification.', """
batches_cover_every_piece_once_in_order|Can a crowded pixel resume without losing or duplicating visible pieces?|Consume a mixed sequence of flat and curved pieces across finite batches and check every emitted ordinal and cumulative slot/block count.
batch_never_splits_a_piece_and_keeps_exact_capacity|Does a full batch cut a visible contribution in half?|Fill exactly800slots and32blocks, then require the next complete piece to begin the next batch.
forged_or_unreachable_prefix_cannot_complete|Can an inconsistent continuation appear to finish a pixel?|Change prefix totals or advance beyond reachable pieces and require explicit refusal before completion.
invalid_extent_and_cursor_refuse|Can malformed pieces or a seventeenth batch bypass the work ceiling?|Reject zero/oversized extents, invalid ordinals and out-of-range batch indices.
sixteenth_batch_can_complete_but_cannot_yield_again|Is the total pixel ceiling actually finite?|Accept completion exactly at12800slots and512blocks, but refuse one additional piece.
empty_stream_has_one_complete_receipt_and_no_emission|Does empty coverage remain distinguishable from an unfinished pixel?|Require a complete zero-count receipt and no emitted contributions.
""")
_add('packaging/test_opaque_batch_geometry','Research native CPU geometry and first-derivative controls. Exact areas and polynomial motion do not qualify general nonlinear Principled shading or complete frames.', """
streamed_pixel_preserves_all_faces_and_closed_unit_area|Can the crowded pixel keep every face while reusing a small sample buffer?|Stream a32-face fan, check owner/padding and barycentric fields, and compare its complete covered area to the unit square.
streamed_geometry_rejects_changed_prefix_before_acceptance|Can a changed saved count silently skip part of a pixel?|Alter the first batch slot total and require native continuation refusal.
streamed_empty_pixel_is_complete_and_zero|Does a pixel outside the mesh finish with unambiguous empty output?|Check complete zero counts, negative owner padding and zero sample data.
streamed_pixel_keeps_candidate_refusal|Does integration streaming accidentally raise the separate face-candidate cap?|Send33candidate faces and require refusal under the unchanged32-face limit.
streamed_material_translation_retains_signed_and_tiny_boundary_flux|Does batching preserve the direction needed to fit moving geometry?|Translate a material-bearing32-face mesh and compare with the exact integral and derivative, including positive and negative1e-20 contributions.
streamed_curved_horizon_has_closed_area_and_boundary_motion|Can a curved shading boundary cross several batches without losing its motion contribution?|Integrate seven disjoint conic-cut triangles and compare area, complement and two boundary motions to high-precision closed formulas.
exact_separation_orientation_matches_rational_signs|Can uncertain overlap tests be resolved without dropping tiny real intersections?|Compare error-free expansion signs with exact rational determinants, including collinear, near-collinear and scaled cases; require unknown for unsafe exponents.
""")
_add('packaging/test_opaque_batch_cuda','Research CUDA kernel conformance only. Device agreement supplements independent analytic CPU checks; this is not an installed-wheel or nonlinear frame qualification.', """
streamed_cuda_matches_cpu_samples_and_first_derivatives|Do CPU and GPU preserve the same arithmetic across batch boundaries?|Compare every owner/sample byte and all vertex and curved-cut first-derivative contractions on crowded flat and curved fixtures.
""")
_add('packaging/test_opaque_batch_protocol','Research host continuation protocol. Input hashes are in-process consistency guards, not authentication; production shading, autograd and allocation admission remain separate.', """
stream_completion_is_atomic_and_summation_retains_order|Can a partial pixel escape or batching change the reduction order?|Keep results unavailable until completion and carry a cancellation-sensitive signed sample sequence in its original order.
changed_inputs_and_cancelled_streams_cannot_resume|Can a resumed pixel mix geometry versions or continue after cancellation?|Mutate input bytes and cancel a separate stream; both must permanently refuse continuation and results.
stream_detects_mutation_after_begin|Can inputs change between issuing a batch and accepting its receipt?|Modify geometry after begin and require refusal when the batch returns.
inconsistent_receipt_poisoning_is_terminal|Can contradictory counts produce a plausible complete pixel?|Alter ordinals, prefixes, extents, completion flags and work counts; require terminal refusal.
duplicate_receipt_cannot_replay_a_contribution|Can the same batch be accumulated twice?|Resubmit an already consumed cursor and require a poisoned incomplete result.
sixteenth_partial_batch_and_shared_work_deadline_refuse|Do continuation boundaries reset time or work allowances?|Reject an unfinished sixteenth batch, an expired shared deadline and excessive aggregate clipping work.
partial_or_nonfinite_sample_payload_is_refused|Can missing samples or dirty padding become accepted evidence?|Inject a missing owner, nonzero padding or nonfinite sample and require refusal before exposing a pixel.
protocol_consumes_real_native_batches_before_exposing_pixel|Do the native producer and guarded consumer agree on actual continuation receipts?|Run the32-face mesh through both layers and require the complete closed area and linear-colour integral after multiple batches.
""")

_add('python/test_native_opaque_batch','Installed C/Torch continuation controls. Independent simple integrals and device bytes qualify these fixtures, not arbitrary nonlinear scenes or full frames.', """
public_batch_area_and_material_translation|Does the installed API preserve a crowded pixel and its movement across batches?|Integrate a32-face fan at ordinary and tiny signed scales; compare unit area, linear colour and translation derivative with exact formulas.
public_batch_prefix_budget_and_first_only_guards|Can a forged prefix or excessive request bypass the bounded continuation contract?|Change cumulative counts, request a seventeenth batch or insufficient workspace, and request a second derivative; each must refuse.
public_batch_input_mutation_is_detected_before_backward|Can backward silently use different geometry from forward?|Modify a saved vertex tensor in place and require PyTorch version checking to refuse the derivative.
public_batch_cpu_cuda_bytes_and_first_derivatives|Do the installed CPU and GPU APIs agree across every batch?|Compare all owners, sample values, vertex derivatives and receipt fields byte for byte on a crowded analytical fan.
public_batch_curved_area_and_motion_have_closed_reference|Are curved boundaries and their movement preserved by the installed continuation API?|Integrate seven disjoint curved-cut triangles over four batches; compare full area, shaded-side area and two cut motions against high-precision closed formulas.
""")
_add('python/test_opaque_pixel_batches','Installed nonlinear scene continuation controls. Equivalent-surface and finite-difference comparisons supplement independent reference tests; no complete original-frame acceptance is implied.', """
scene_batch_completes_overflow_and_matches_equivalent_surface|Can a pixel rejected by the resident renderer finish through bounded streaming?|Render a32-face surface with the installed production dispatcher and compare its opaque image to an equivalent two-triangle surface.
scene_batch_first_vjp_replays_without_retaining_sample_graphs|Can fitting differentiate a crowded pixel without storing every shading graph?|Inspect saved tensors for sample buffers, then compare replayed albedo derivatives with the equivalent simple surface.
scene_batch_checks_input_versions_and_rejects_second_derivatives|Does production replay enforce its input and derivative contract?|Change a material tensor after forward and separately request second derivatives; both operations must fail explicitly.
scene_batch_background_is_composited_once|Does each batch accidentally add another copy of the background?|Check a fully covered pixel against the background-free result and require a zero background derivative after all batches.
scene_batch_coloured_material_motion_matches_two_differences|Does moving the material-bearing mesh retain a meaningful optimization direction?|Translate a crowded surface with varying vertex albedo and compare the nonzero first derivative with two centered differences.
""")
_add('python/test_opaque_pixel_reference','Independent interval evidence for a smooth full-Principled pixel and albedo tangent. This does not cover arbitrary horizons, original-frame pixels or frame execution time.', """
crowded_scene_matches_independent_nonlinear_value_and_albedo_tangent|Does the crowded production pixel agree with an independent numerical reference?|Enclose the continuous surface integral and albedo tangent at1e-12radius, compare actual installed CPU/CUDA rendering under the unchanged tolerances, and require exact device bytes separately.
""")

_add('python/test_native_opaque_batch','Installed boundary-topology regressions. Exact analytical moments expose derivative errors that a single colour field can miss; general scene and frame admission remains separate.', """
public_batch_translation_preserves_closed_area_and_both_moments|Can a moving mesh invent coverage change even while the forward image looks correct?|Translate the full-cover fan and compare constant, X and Y material moments in both axes under material-following and cut-bearing rules, using full autograd cotangents.
""")

_add('python/test_native_opaque_batch_workspace','C ABI metadata and workspace qualification only. Query success allocates nothing and does not certify a renderer result or device performance.', """
material_workspace_queries_match_scratch_and_reject_invalid_metadata|Can a C caller size the new operation scratch without relying on a different entry point?|Resolve both installed material workspace queries, compare float32/64 size and alignment with the shared scratch contract, and reject missing outputs, invalid dtype and reserved metadata.
""")

_add('packaging/test_opaque_disjoint_visibility','Exact rational visibility controls. Zero-area overlap may avoid fragmentation; every positive overlap, including tiny slivers, remains represented. Shading accuracy requires separate evidence.', """
disjoint_or_touching_occluder_preserves_original_polygon|Can unrelated triangles needlessly fragment the independent reference?|Prove zero intersection area before preserving the original polygon, including boundary-only contact.
positive_overlap_including_tiny_sliver_keeps_exact_moments|Could compaction erase a tiny real contribution?|Subtract ordinary and extremely small positive intersections and compare exact rational area and first moments.
compact_visible_cover_has_exact_per_face_area_and_moments|Does compact visibility preserve which face owns each contribution?|Compare per-face area and moments between historical and compact exact decompositions.
disjoint_proof_charges_the_existing_work_budget|Does proving separation create uncounted geometric work?|Require the exact overlap proof to consume the caller's existing clipping budget and refuse exhaustion.
""")
_add('packaging/test_opaque_shared_pixel_reference','Independent whole-pixel interval controls. All visible pieces share one error and work pool; closed scenes qualify the mechanism, not arbitrary original-mesh pixels or frames.', """
shared_pixel_pool_matches_closed_piecewise_constant_integral|Does pooling independent pieces preserve the full shaded image?|Compare a two-material-region pixel against its closed piecewise constant Principled integral with outward intervals.
shared_pool_closures_keep_distinct_piece_fields_and_charge_one_pool|Could one piece accidentally reuse another piece's material calculation?|Inspect distinct prepared fields, zero depth origins, one global tolerance and one work pool.
shared_reference_preserves_legacy_default_and_refuses_campaign_mixing|Can the new optional reference reinterpret historical accuracy trials?|Compare legacy and shared enclosures on a closed scene and reject mixing a selected-piece campaign with the shared mode.
""")
_add('packaging/test_opaque_disjoint_roots','Reference accounting for independently certified disjoint physical pieces. Geometric coverage is the caller obligation; all pieces must finish under one unchanged global work/error pool.', """
independent_roots_preserve_closed_integral_tangent_and_actual_work|Should grouping independent pieces consume their subdivision depth?|Start each physical piece at depth zero, charge both evaluations, and verify the closed total integral and derivative.
independent_roots_never_accept_partial_cover_or_reset_budget|Can a successful first piece hide an unresolved second piece?|Refuse insufficient total work and an unresolved later root without returning a partial pixel as accepted.
independent_root_contract_rejects_ambiguous_accounting|Can incompatible depths or schedules bypass the accounting contract?|Reject nonboolean mode flags, missing covers, inherited depths and saved partition schedules.
default_chart_partition_still_requires_complete_binary_depths|Does the new mode weaken historical partition validation?|Keep the original binary-cover depth check and internal-node work charges on the default path.
""")

_add('packaging/test_opaque_shared_pixel_reference','Optional whole-pixel reference integrations preserve the historical default. Tighter quadratic ranges and aggregate angular bounds must remain conservative; original-scene acceptance is separate.', """
shared_curved_reference_tight_root_preserves_field_and_first_jet|Can the exact quadratic bound remove a false horizon-domain failure without changing shading?|Compare the prepared full material fields through derivative order10 with the original algebra on a curved fixture and reject ambiguous option values.
shared_aggregate_residual_keeps_distinct_fields_and_complete_bounds|Can pieces share expensive angular-table preparation safely?|Require shared and separately prepared tables to produce identical complete correction endpoints on distinct piece fields; reject use outside the explicit shared-pixel mode.
""")

_add('packaging/test_opaque_shared_pixel_reference','Localized smooth proof retains the complete pixel and its existing resource limits.', """
shared_localized_smooth_remainder_keeps_one_budget_and_closed_answer|Does the sharper smooth proof preserve the complete pixel contract?|Check the shared256-cell/depth10/5e-13 pool, every chart configuration, exact piecewise-constant answers and rejection of invalid or implicit profiles. Knot corrections remain separately included; this control alone does not admit the original pixel.
""")

_add('packaging/test_opaque_shared_pixel_reference','An alternative smooth base is valid only together with its full signed material correction.', """
shared_stable_view_base_preserves_full_value_and_spatial_jets|Does avoiding near-horizon logarithmic extrapolation change the material?|Compare the sum of smooth and correction terms through third spatial derivatives in both directions. Require the conditional branch choice, shared aggregate profile and explicit Boolean selection; retain all residual contributions.
""")

_add('packaging/test_opaque_shared_pixel_reference','Exact derivative cancellation can sharpen monotonicity proofs without dropping terms.', """
shared_tight_light_slope_retains_complete_spatial_coefficients|Does the light-slope refinement retain the full field?|Require the first spatial coefficient to be a contained enclosure, preserve every other spatial coefficient exactly and match full/material-only role queries. Reject implicit or malformed opt-ins; complete pixel qualification remains separate.
""")

_add('packaging/test_opaque_cosine_slopes','Exact rational cancellation precedes interval evaluation; all coordinate derivatives remain included.', """
cancelled_slopes_enclose_closed_normalization_and_tiny_terms|Do cosine slopes match an independent closed expression?|Compare both angular roles and both spatial directions to analytic normalized-vector formulas, including2^-60 coefficients that must remain positive.
cancelled_slopes_reject_intrinsic_motion_and_bad_contracts|Can a fixed-field proof silently omit motion?|Reject nonzero intrinsic field motion, invalid roles and insufficient or mismatched spatial orders.
""")

_add('packaging/test_opaque_cosine_slopes','A finite interval cover strengthens the derivative proof without sampling away uncertainty.', """
subdivided_slopes_cover_every_box_and_preserve_chain_and_tiny_terms|Do smaller derivative boxes cover the entire region?|Check all eight boxes, exact total area, both signed coordinate derivatives, independent rational derivative values and a retained2^-60 contribution.
subdivided_slopes_reject_unbounded_work_and_propagate_cancellation|Can derivative subdivision exceed its bounds or ignore cancellation?|Reject more than eight boxes and malformed partitions; require immediate propagation of the worker checkpoint interruption.
""")
_add('packaging/test_opaque_streamed_reference_contract','One explicitly approved22-piece offline campaign. Certificates bind exact geometry, conserved error shares and successful bounded workers; partial pixels never pass.', """
streamed_input_and_allocation_bind_all22_pieces|Can per-piece work silently change the pixel or its accuracy budget?|Reject changed geometry and ordering; check every positive downward area share and their total against the unchanged pixel radius.
aggregation_keeps_every_positive_tiny_piece_and_exact_order|Can tiny pieces disappear when certificates are combined?|Sum22 positive2^-60-scale contributions outward and compare with their exact rational total.
aggregation_rejects_incomplete_or_unbound_evidence|Could a failed or mismatched certificate qualify the pixel?|Reject missing, duplicate, reordered, altered-identity, nonfinite, over-radius and over-budget certificates, including failed monitored processes.
frozen_sources_are_checked_as_bytes|Does a recorded filename establish checker identity?|Hash actual source bytes and reject a change at the same path.
streamed_profile_keeps_limits_and_refuses_mixing|Does the offline allowance change production or historical profiles?|Check unchanged per-worker limits, strict one-piece selection and exact allocated tolerance; refuse implicit selection and mixed campaign modes.
""")
_add('packaging/test_opaque_streamed_reference_campaign','Campaign scheduling controls use synthetic workers; actual original-piece numerical acceptance is recorded separately.', """
campaign_is_serial_single_use_and_stops_first_failure|Will the runner continue after a failed piece or reuse old evidence?|Simulate a third-piece failure, require an immediate stop without aggregation and refuse reuse of the campaign directory.
campaign_aggregates_only_after22_successes_and_rechecks_identity|When may a complete pixel be compared?|Require all22 serial successes before the aggregation worker and check frozen identity around each launch.
changed_manifest_stops_before_next_worker|Can a changed checker be mixed into a running campaign?|Change the identity result after one worker and require the controller to stop before another launch.
preserved_device_comparison_checks_complete_values_and_derivatives|Does device agreement alone admit accuracy?|Compare all RGB and albedo derivatives to independent bounds; reject byte mismatches and flag an out-of-bound derivative.
""")
_add('packaging/test_opaque_streamed_extended','User-approved offline-only work extension. The frozen pixel, conserved accuracy budget, memory, serial execution and all rendering limits remain unchanged.', """
extended_streamed_profile_requires_explicit_mode_and_keeps_old_limits|Can a larger checker budget silently affect production?|Require the explicit extended mode, verify its exact15-minute and4096-cell limits, and retain historical defaults.
extended_streamed_quadrature_keeps_accuracy_and_finite_work|Does allowing more work weaken the numerical answer?|Compare a closed polynomial integral and derivative; reject looser accuracy, excess caps and unresolved work beyond its budget.
extended_supervisor_restores_standard_limits_after_worker|Can an extended checker leak its limits into later workers?|Inspect exact piece and aggregation deadlines, then require restoration of the original supervisor configuration even after failure.
extended_campaign_runs_failed_piece_first_then_fresh_remaining22|Does scheduling target the known blocker while retaining complete coverage?|Require piece1first, then0and2through21exactly once, followed by aggregation only after all22succeed.
extended_campaign_first_failure_launches_nothing_else|Will a failing first probe trigger hours of unnecessary work?|Fail the previously problematic piece and require no later piece or aggregation launch.
extended_certificate_uses_larger_work_limits_but_same_error_share|Can a larger allowance accept an inaccurate certificate?|Keep identical error allocations while validating expanded work limits; reject excess time, memory, cells and interval width.
extended_full_piece_binding_preserves_allocation_and_charges_deadline|Does the actual material checker honor the approved profile?|Inspect the exact piece allocation and4096-cell/depth14arguments; simulate deadline exhaustion and require explicit refusal.
""")
_add('packaging/test_opaque_streamed_hour','Approved one-hour offline checking changes time only. The4096-cell/depth14 work cap,1GiB memory,exact allocations,numerical criteria and rendering limits remain unchanged.', """
hour_profile_changes_only_time_and_refuses_other_modes|Can the time extension leak into another workload?|Compare every non-time limit with the15-minute profile and reject implicit or mismatched worker modes.
hour_certificate_keeps_error_allocation_and_rejects_excess|Can longer checking relax its proof obligation?|Require identical input,geometry and error shares; reject excessive work,time,memory or uncertainty and mixing historical-profile certificates.
hour_full_piece_and_quadrature_enforce_time_work_and_accuracy|Does the numerical checker use the approved finite limits?|Verify a closed integral,refuse loose accuracy and excessive work,then simulate expiration of the actual piece checker at3600seconds.
hour_controller_keeps_first_failure_order_and_restores_worker_limits|Does longer checking retain serial scheduling and bounded aggregation?|Require piece1first and every other piece once; inspect3600-profile supervision,standard aggregation limits and restoration after a failed worker.
""")

_add('packaging/test_opaque_centered_correction','Centered C2 derivative enclosures can recover lost monotonicity without changing the integrand. These controls qualify the proof path; full original-piece acceptance remains separate.', """
centered_correction_contains_independent_closed_log_integral|Does the repaired correction contain an independently integrated answer?|Compare four signed curved cases with a closed logarithmic rectangle integral and verify restoration of the original geometry hooks.
centered_correction_falls_back_only_after_restoring_original_inputs|Can a legitimate proof refusal corrupt the historical fallback?|Inject each allowed domain refusal, verify restored inputs and integration order, and retain tiny positive and negative contributions.
centered_correction_propagates_invalid_proof_and_cancellation|Could cancellation or inconsistent evidence silently become acceptance?|Require both failures to propagate without invoking fallback, while restoring all temporary geometry hooks.
centered_correction_is_explicit_and_preserves_streamed_obligation|Can tighter bounds change accuracy or another workload implicitly?|Require explicit streamed selection, preserve the exact area allocation and work limits, and reject malformed or implicit modes.
centered_campaign_requires_explicit_frozen_mode_and_keeps_old_order|Does the repair require a named frozen candidate?|Reject unknown or mismatched checker modes; keep historical scheduling and select the currently failing piece first only for the new mode.
centered_campaign_checks_failed_piece_first_without_reusing_certificates|Can a new candidate reuse old successes or continue after failure?|Simulate serial completion of all22fresh pieces and aggregation, stop immediately on first failure, and reject reused campaign directories.
""")

_add('packaging/test_opaque_directional_correction','The explicit directional checker selects between complete existing proof identities. Local controls do not certify an original complete piece, pixel or frame.', """
directional_correction_contains_closed_integral_and_restores_hooks|Does direction selection preserve the mathematical integral?|Compare four signed curved cases with an independent closed logarithmic integral and require restoration of original geometry hooks.
directional_many_knots_keep_tiny_discontinuous_and_product_terms|Can wider angular regions lose tiny or simultaneous jumps?|Use sixteen discontinuous branches in both directions, with values of order1e-30, and compare single and simultaneous cosine changes with exact piecewise integrals.
directional_invalid_evidence_and_cancellation_never_fall_back|Can invalid proof evidence or cancellation become a passing fallback?|Inject both failures and require propagation with original inputs restored and no fallback call.
directional_domain_refusal_restores_fallback_and_tiny_bounds|What happens when neither integration direction is supported?|Require the historical fallback after hook restoration and preserve both endpoints of a tiny signed enclosure.
directional_mode_preserves_exact_piece_obligation|Does the optional repair change work or accuracy limits?|Require explicit centered streamed selection, unchanged exact piece allocation and4096-cell/depth14arguments; reject implicit and malformed switches.
directional_campaign_is_frozen_fresh_serial_and_stops_first_failure|Can the new checker mix old certificates or keep running after failure?|Require the named frozen mode and piece3first, all22fresh serial successes before aggregation, immediate stop on failure and refusal of reused output directories.
""")

_add('packaging/test_opaque_horizon_correction','A complete aggregate identity handles the first angular knot at the horizon. Independent analytical controls and saved-region comparisons qualify this checker locally; they do not admit a complete original piece, pixel or frame.', """
horizon_correction_contains_exact_tiny_piecewise_integrals|Can the first horizon crossing lose tiny or simultaneous contributions?|Compare both coordinate directions and signs against exact rational piecewise integrals, including discontinuous branches and simultaneous view and light changes at tiny nonzero scales.
horizon_log_curvature_contains_independent_antiderivative|Does the horizon repair contain independently integrated nonlinear shading?|Compare both directions and signs with a separately evaluated logarithm-squared antiderivative using100-digit Decimal arithmetic.
horizon_invalid_evidence_and_cancellation_propagate|Can an invalid proof or cancelled worker silently pass through fallback?|Inject inconsistent evidence and cancellation and require propagation while restoring the original geometry hooks.
horizon_domain_refusal_restores_historical_fallback|What happens when neither aggregate direction has a valid monotonicity proof?|Refuse both axes, restore the original inputs, and preserve the historical fallback including both tiny signed endpoints.
horizon_selection_leaves_nonhorizon_domain_unchanged|Does a horizon repair alter unrelated shading regions?|Verify that a nonhorizon region retains the historical directional path and its tiny nonzero result.
horizon_mode_preserves_exact_piece_obligation|Can the repair change the error budget or work limits?|Require the explicit streamed centered directional mode, preserve exact area allocation and4096-cell/depth14limits, and reject malformed or implicit selection.
horizon_frozen_campaign_keeps_serial_fresh_stop_policy|Can the new checker reuse certificates or keep going after failure?|Require the named frozen mode, piece3first, all22fresh serial successes before aggregation, immediate stop on failure and refusal of a reused output directory.
""")

_add('packaging/test_opaque_root_contraction','Experimental crossing geometry controls. These qualify mathematical enclosures on analytical fixtures; the prototype is not adopted for full campaigns because representative cost is worse and complete-piece adequacy remains unproved.', """
contracted_crossing_contains_independent_quadratic_root|Does finite root contraction still contain the true crossing?|Compare positive and negative curved crossings, including1e-40scales, with an independent100-digit quadratic root.
contracted_clipped_crossing_retains_absence_and_endpoints|Can clipping discard a boundary contribution when the crossing is absent?|Check exact affine roots inside,outside and at both endpoints in both directions; preserve the clipped endpoint.
contracted_crossing_keeps_uncertain_knots|Does uncertain knot location retain every possible root?|Require the full interval of exact roots for uncertain targets in both crossing directions.
contracted_crossing_refuses_invalid_contract_and_propagates_cancel|Can the helper run unboundedly or ignore cancellation?|Reject invalid iteration counts and clipping flags,propagate cancellation,and verify certified absence outside the domain.
contracted_material_contains_independent_log_integral|Does recomputing jump coefficients on contracted supports preserve the full correction?|Compare four signed curved material cases against an independent closed logarithmic integral and verify geometry-hook restoration.
""")

_add('packaging/test_opaque_streamed_two_hour','The approved opt-in offline checker can spend more time and region evaluations. These controls do not loosen image accuracy, rendering limits or certify the original pixel.', """
two_hour_profile_changes_only_time_and_refuses_other_modes|Can ordinary rendering inherit the larger offline allowance?|Compare unchanged storage and depth with the old profile and reject mismatched worker modes; only the named offline mode receives the new limits.
two_hour_certificate_keeps_error_allocation_and_rejects_excess|Can a longer run pass with a weaker certificate?|Preserve exact input and area-allocation identities and reject excessive work,time,memory,interval width or a certificate from another profile.
two_hour_full_piece_and_quadrature_enforce_time_work_and_accuracy|Are the new limits actually enforced during integration?|Check an exact polynomial integral, reject excessive depth or cells and weak accuracy, then inject an overdue clock and require the complete piece to stop.
two_hour_controller_keeps_first_failure_order_and_restores_worker_limits|Does the supervisor keep aggregation small and restore normal limits?|Inject worker failure, inspect piece and aggregation deadlines and memory, and require restoration of the original supervisor profile.
two_hour_campaign_fresh_serial_identity_and_stop|Can the expanded campaign reuse results or continue after failure?|Simulate all22serial workers before aggregation, reject reused output, stop at the first failed worker and stop after candidate identity changes.
two_hour_combined_allocation_requires_all_fresh_pieces|Can partial or mismatched certificates admit the whole pixel?|Verify aggregate accounting at the approved ceilings and reject missing,reordered,duplicated or differently bound certificates; numerical thresholds remain unchanged.
""")


_add('python/test_opaque_frame_culling','One-pixel installed CPU/CUDA fixtures qualify frame visibility identity and replay. No original-mesh accuracy, complete-frame throughput or scale admission follows.', """
explicit_frame_culling_matches_installed_pixels_and_vjp|Can saving and replaying a frame silently change which face orientations are visible?|Exercise all three explicit culling modes, compare every saved pixel and vertex-gradient byte with the same installed mode, and reject altered receipt identity before replay renders.
frame_culling_invalid_modes_refused_before_storage|Can an invalid culling mode create partial frame storage?|Reject boolean, out-of-range, floating and string modes before creating the output directory.
default_frame_culling_preserves_legacy_receipt|Does adding explicit culling change an existing caller's frame contract?|Run the original no-culling default, require the unchanged version-one receipt identity, and replay its pixels successfully.
""")


_add('packaging/test_opaque_remaining_reference','Only the two frozen remaining original128 pixels. Synthetic certificates and controller simulations test safeguards, not image correctness or full-frame performance.', """
remaining_profile_is_explicit_and_preserves_per_piece_limits|Could ordinary rendering inherit the two-pixel offline budget?|Require the exact profile and worker mode, compare all numerical and resource limits, and reject unrelated modes.
remaining_allocation_and_complete_certificates_bind_each_input|Can missing pieces or mixed input certificates admit either complete pixel?|Conserve the full error allowance across33 or28 positive shares, aggregate every fresh certificate, and reject missing, reordered, altered or excessive evidence.
remaining_full_piece_selector_cap_and_deadline_are_enforced|Can a new profile silently enlarge the default checker or ignore its deadline?|Reject invalid selectors and another pixel, retain the default32-piece refusal, and inject an overdue clock into full-piece integration.
remaining_controller_serial_complete_and_first_failure|Can a two-pixel piece controller continue after failure or reuse output?|Simulate each serial piece before aggregation, inspect worker ceilings and restored limits, refuse reused output, and stop the first failure.
remaining_quadrature_keeps_independent_integral_and_refusals|Does the new offline name alter the numerical acceptance rule?|Integrate an exact polynomial with known quarter-area integral and reject excessive work, depth or weaker accuracy.
remaining_pair_campaign_is_bounded_single_use_and_stops|Could failure in the first pixel still launch the second pixel?|Simulate both complete controllers in their declared order, refuse reused output, and require immediate stop on first pixel failure.
remaining_manifest_rejects_changed_inputs_scope_and_limits|Can changed bytes, omitted resources or another profile enter the trial?|Load the frozen manifest, then mutate inputs, scope, allocation, limits and hashes and require rejection before numerical execution.
remaining_comparison_selects_correct_preserved_device_case|Can another sample pixel be used as the CPU/CUDA comparison?|Use deliberately different case bytes to require the declared case and reject another case or out-of-bound colour values.
historical_contract_cannot_mint_remaining_profile_for_old_input|Can the old22-piece helper label its input as one of the new pixels?|Try both new profile names with the historical input and require refusal, preserving the old certificate boundary.
""")


_add('packaging/test_opaque_light_partition','Exact fixed-geometry light support for the two new frozen pixels; it does not certify moving light-boundary derivatives or renderer speed.', """
light_partition_retains_tiny_positive_area_and_exact_complement|Can light clipping erase a very small illuminated contribution?|Compare the exact rational lit area with a closed formula at ordinary and extremely small positive scales and require exact lit-plus-dark area conservation.
light_partition_dark_lit_crossing_and_orientation|Does the partition distinguish exact darkness from a light boundary consistently?|Check dark, zero, lit and crossing triangles in both orientations against independent rational areas and bounded component counts.
light_partition_perspective_sign_and_cancel|Are light signs computed before normalization with the right perspective weights?|Check a nonunit perspective-weight triangle against its exact area, reject invalid clip weights and propagate cancellation.
light_partition_nonlinear_integral_has_independent_log_answer|Does clipping preserve a complete nonlinear illuminated integral?|Integrate the clipped rational kernel and compare its full enclosure with the independent closed answer three-quarters minus log two.
remaining_light_partition_charges_all_components_and_certifies_darkness|Can split pieces reset the error or work budget?|Verify both lit components consume the same original cell pool and conserved error share, while a proved dark piece returns exact zero without numerical quadrature.
""")

_add('packaging/test_opaque_report_io','Reporting reliability only. Retries publish the same bytes and remain charged to the existing worker clock; numerical work is never retried or accepted by this mechanism.', """
transient_windows_lock_retries_same_report_without_recomputing|Can a briefly locked report destroy completed numerical work?|Inject three known Windows lock errors, release each lock, and require identical report bytes across bounded replacement attempts.
persistent_lock_is_bounded_and_preserves_previous_report|Can reporting retry indefinitely or overwrite the last valid evidence with partial bytes?|Keep the lock held, require seven attempts and at most0.63seconds of requested waiting, preserve the previous report and retain the new temporary bytes.
unrelated_io_failure_is_not_retried|Are ordinary permission and unrelated I/O errors hidden?|Inject an unrelated permission error and require immediate failure without any retry delay.
actual_windows_reader_lock_release_allows_atomic_publication|Does the repair work with a real Windows file handle?|Hold a real non-delete-sharing reader handle, release it after a bounded delay, and require successful complete JSON publication.
""")

_add('packaging/test_opaque_weighted_light','Experimental ordinary-derivative bound only; complete-piece success and campaign adoption remain separate. All knot jumps are retained by the existing independent jump bound.', """
weighted_residual_contains_independent_log_cubic_derivatives|Does keeping the light factor attached preserve every ordinary derivative?|Compare value and first three spatial Taylor coefficients against independent90-digit formulas for light-weighted log cubics, with linear-horizon and ordinary base branches.
weighted_horizon_zero_and_invalid_profile|Can the weighted bound lose a tiny or unsupported contribution?|Require exact zero only on the identical horizon branch and refuse negative light ranges or unsupported derivative orders.
weighted_material_keeps_complete_independent_step_product_integral|Are simultaneous view and light changes still included?|Compare both ordinary-bound modes with the independent complete integral of16discontinuous steps and their product, retaining all jump terms.
""")
_add('packaging/test_opaque_depth_refusal','Checker cost safeguard, not improved image accuracy. Exact fixed-leaf interval widths prove when further permitted subdivision cannot pass the unchanged total bound.', """
depth_fixed_width_stops_before_refining_other_charts|Can the checker waste work after its depth cap makes success impossible?|Give one immutable leaf an excessive interval and require immediate refusal with the complete saved cover before refining another chart.
opposite_signed_fixed_values_cannot_cancel_interval_width|Can opposite interval centers hide irreducible uncertainty?|Use equal opposite centers with nonzero widths and require refusal; uncertainty widths add rather than cancel.
depth_fixed_uncertainty_below_whole_allowance_still_passes|Does the cost safeguard impose a stricter local accuracy quota?|Allow a depth-limited uncertain contribution that fits the whole allowance and require the unchanged complete-sum pass.
""")
_add('packaging/test_opaque_light_map','Experimental fixed-geometry coordinate change, not a new renderer model, extra subdivision budget or completed pixel certificate.', """
light_map_preserves_area_and_polynomial_moments|Does endpoint flattening preserve the whole integration domain and its Jacobian?|Compare the complete mapped square area and first polynomial moments with exact independent integrals.
light_map_complete_nonlinear_integral_matches_log_two|Does the coordinate change preserve a nonlinear integral?|Compare the complete mapped rational integral with independent outward log two under the unchanged reference tolerance.
light_map_keeps_tiny_support_and_rejects_undeclared_power|Can the map truncate small positive regions or silently change its policy?|Keep coordinates of1e-40and their Jacobian strictly positive, refuse undeclared powers and preserve the identity map.
full_checker_map_is_explicit_and_matches_change_of_variables|Are the full checker values and spatial coefficients connected consistently?|Require explicit mapped-profile selection and compare the original mapped field times its exact Jacobian with the connected checker through spatial order3at three points.
""")

_add('packaging/test_opaque_light_map','Explicit checker-coordinate selection; this does not grant a complete-piece certificate.', """
light_map_leaves_fully_lit_pieces_on_original_coordinates|Does the repair unnecessarily change pieces away from the light boundary?|Require the original and opt-in checker fields to retain identical interval endpoints through spatial order3on an actual fully lit piece.
""")
_add('packaging/test_opaque_remaining_reference','Frozen mapped candidate routing and order only. Actual numerical certificates remain required for every piece.', """
mapped_candidate_is_explicit_and_runs_failed_piece_first|Does the new trial select the exact repair and preserve all pieces?|Keep historical options/order unchanged, require explicit map4and weighted-residual flags, start with previously stalled piece9, retain all28or33indices exactly once, and reject unknown candidates.
""")


_add('packaging/test_opaque_light_alignment','Independent checker coordinates only: these tests preserve the exact integration domain and derivatives, but do not certify a complete original pixel or production speed.', """
alignment_preserves_every_corner_area_orientation_and_moments|Does boundary alignment preserve every triangle orientation and contribution?|Exercise all six corner permutations with one or two zero-light corners; require the same vertices, signed area and independent exact area, first and second moments.
alignment_keeps_fully_lit_order_and_tiny_positive_support|Can alignment move unaffected triangles or discard tiny light values?|Keep fully lit triangles in their original order and preserve exact positive light values as small as one over ten to the eightieth power.
alignment_rejects_nonpositive_or_unclipped_support|Can an invalid triangle be treated as certified illuminated support?|Reject identically dark and sign-crossing triangles before applying the positive-support coordinate map.
aligned_original_fields_preserve_physical_values_and_spatial_derivatives|Does the connected checker still integrate the same physical shading field?|For every lit component of saved pieces9,10,11and16, independently transform barycentric coordinates and the Jacobian and compare field values and three spatial derivative coefficients at two points.
alignment_requires_explicit_remaining_mapped_profile|Can the new coordinate option silently affect a legacy checker?|Reject nonboolean flags and alignment without the explicit remaining-pixel power-four map.
""")
_add('packaging/test_opaque_remaining_reference','Candidate and order controls only; all fresh piece certificates and complete-pixel comparisons remain required.', """
aligned_checker_is_explicit_and_prioritizes_unqualified_boundary_placements|Does the next campaign exercise previously missed boundary placements before repeating expensive cases?|Require explicit alignment, mapping and weighted-bound options; run pieces10,11and16first while retaining all28indices exactly once and the unchanged33-piece order.
""")


# Original-frame admission checks are controls, never substitute rendered evidence.
_FRAME_ADMISSION_GUIDES = {
'only_approved_startups_need_no_timing_anchors': ('Which frames may start without prior timing anchors?', 'Admit only the approved original128and256square profiles and keep accuracy/storage admission separate.'),
'large_frame_requires_two_matching_complete_anchors': ('Can a larger image launch from missing timing evidence?', 'Require two matching complete lower-resolution frames and check the slower-rate forecast with twice-cost margin plus30seconds.'),
'forecast_must_fit_remaining_whole_worker_time': ('Does setup consume the rendering deadline?', 'Reject a forecast equal to or greater than the actual remaining whole-worker time.'),
'profiles_cannot_borrow_startup_exception': ('Can another model or gradient scope borrow the original-mesh exception?', 'Change coverage,material,precision,geometry,outputs,device or gradient profile and require refusal.'),
'invalid_extents_and_time_refuse': ('Can malformed sizes or deadlines bypass admission?', 'Reject unsupported extents,booleans,nonfinite values and time beyond120seconds.'),
'forward_profile_and_cuda_are_explicit': ('Are CPU/CUDA and forward/backward qualification distinct?', 'Require an explicit supported device and empty gradient scope for a forward-only profile.'),
'larger_frame_preserves_identity_checks': ('Can timing from a different binary justify this frame?', 'Change an anchor wheel hash and require rejection before rendering.')}
for _name,(_purpose,_method) in _FRAME_ADMISSION_GUIDES.items():
    GUIDES['tests/python/test_opaque_original_frame_admission.py::AdmissionTests::test_'+_name]=dict(purpose=_purpose,method=_method,limits='Metadata admission controls only. Actual complete-frame accuracy,resource and time evidence remain required.')
_add('python/test_opaque_original_frame_prerequisites','Hash-pinned reviewed evidence only; this does not prove producer correctness or certify every full-frame pixel.', """
all_exact_pixels_and_sources_are_required|Can an omitted difficult pixel be treated as qualified?|Require all three exact representative pixels and83pieces before admission;reject a missing entry.
changed_pixel_bytes_refuse_even_with_green_summary|Can edited evidence keep its old passing status?|Change saved bytes and require a digest mismatch.
failed_or_inconsistent_comparison_refuses|Can a failed,nonfinite or over-tolerance comparison pass the gate?|Mutate comparison status,error or allowance and require refusal.
changed_storage_source_refuses|Can an unqualified storage implementation reuse an old receipt?|Change its source bytes and require refusal.
changed_gate_or_wheel_refuses|Can another receipt or installed artifact borrow this qualification?|Reject mismatched prerequisite and wheel hashes.
duplicate_pixel_scope_or_piece_count_refuses|Can repeated easy evidence substitute for another pixel?|Change the origin or piece count and require the exact three-pixel scope.
""")
_add('python/test_opaque_original_frame_probe','Complete-storage controls only; numerical rendering and performance require separate actual worker evidence.', """
frame_inspection_requires_complete_finite_hashed_streams|What makes a frame eligible for a timing record?|Consume and hash every stored block,verify finite arrays and exact dimensions,and count actual covered pixels.
nonfinite_complete_output_is_not_a_pass|Can a complete file containing invalid pixels count as a good frame?|Store a NaN in a correctly hashed frame and require validation failure.
corrupt_or_incomplete_frame_is_not_an_anchor|Can partial bytes become a successful timing anchor?|Truncate the colour stream and require rejection before an anchor can be produced.
""")

_add('python/test_opaque_frame_windows','Bounded scheduling equivalence on finite fixtures only; original-frame performance and full-scale acceptance are separate.', """
forward_windows_preserve_pixels_and_canonical_replay|Can grouping forward pixels change the image or gradient accumulation order?|Compare all four stored streams byte-for-byte with the single-pixel baseline,then require identical gradients and exactly one ascending-pixel call per backward pixel;reject a changed window receipt.
invalid_window_limits_refuse_before_storage|Can a caller silently exceed the qualified window size?|Reject noninteger,boolean,zero,negative and over16pixel requests before creating output storage.
""")

_add('python/test_opaque_original_frame_admission','Scheduling identity controls only; actual numerical and performance qualification remain required.', """
windowed_profile_requires_explicit_bounded_window_scope|Can a grouped execution borrow another scheduling profile silently?|Require the explicit version3identity and an integer forward-window cap from2to16;reject malformed or out-of-scope caps.
""")

_add('python/test_principled_shared_curve','Shared storage of identical scalar-roughness curves only; unchanged shading equations and group limits. Full-frame speed is a separate gate.', """
shared_curve_matches_dense_values_jacobian_and_vjp|Can avoiding repeated curve copies change shading or fitting gradients?|Compare values, all first derivatives and vector-Jacobian products byte-for-byte against dense repeated curves up to128points and1024light pairs on each device.
shared_curve_rejects_mixed_shapes_invalid_values_and_old_pair_scope|Can compact storage bypass shape, numerical or public operation limits?|Reject mixed dense/shared rows, nonfinite curve data and over-limit groups; retain the old pair-operation shape contract.
shared_curve_storage_bound_is_explicit_and_retains_point_pair_charges|Does sharing a curve erase unrelated memory costs?|Require an explicit shared flag, retain every point/pair derivative charge and preserve the default dense estimate.
""")

_add('python/test_principled_shared_curve','Native empty-output and legacy-provider safety controls; neither is a frame-performance result.', """
shared_curve_empty_outputs_still_validate_curve_domain|Can an empty result conceal invalid shared input?|Require correctly shaped empty outputs while rejecting nonfinite shared curve rows on both devices.
old_provider_refuses_shared_operation_before_reading_row|Can an old CUDA library mistake one curve for many rows?|In a fresh process with the preserved provider, require old dense groups to work and the new operation to refuse before reading shared data.
""")

_add('python/test_opaque_prepared_geometry','Prepared geometry is local to one bounded forward pixel. Every differentiated batch is recomputed; full-frame timing remains separate.', """
pixel_prepares_geometry_once_without_retaining_it_for_backward|Can repeated forward geometry setup be removed without changing the image or gradients?|Count projection calls across a real overflowing pixel, require at most one setup (or reuse of the owning window setup) followed by reuse, require every differentiated batch to prepare anew, and compare complete colour/mask/albedo-gradient bytes with preserved qualified evidence.
""")

_add('python/test_opaque_prepared_geometry','Mutation must abort before another batch can use retained forward geometry.', """
forward_reuse_refuses_input_mutation_between_batches|Can a changed input silently reuse stale forward setup?|Modify a material tensor after the first batch and require an immediate input-version failure before the next evaluation.
""")

_add('python/test_opaque_sparse_horizons','Forward-only exact face selection. Differentiated arithmetic stays dense; extreme inputs retain original validation.', """
forward_horizons_visit_only_selected_faces_and_match_dense|Why calculate horizon coefficients for faces outside this pixel?|Require work on all21 selected faces instead of10000, while every scattered coefficient exactly matches the original dense path.
extreme_finite_inputs_retain_dense_horizon_validation|Can skipping irrelevant faces conceal arithmetic overflow?|Use extreme finite normal values and require fallback to the full original validation path.
""")

_add('python/test_opaque_shared_projection','Only static forward geometry is reused within one public window; selected-region horizon work and dense backward preparation stay separate.', """
window_prepares_static_geometry_once_and_releases_it|Can one forward window avoid repeatedly projecting the same scene?|Require one projection instead of15, exact saved original-window pixels, and release of all projected tensors even while output autograd context remains alive.
window_reuse_refuses_mutation_before_next_tile|Can an input change after preparation yield stale pixels?|Mutate the albedo after one internal tile and require rejection before the next tile is evaluated.
""")

_add('python/test_opaque_ordered_accumulation','Renderer-internal ascending sample carry only; no reassociation, contribution cutoff, derivative change or frame-performance claim.', """
ordered_rgba_retains_each_rounded_add_and_partition|Does the native carry preserve every separately rounded addition?|Compare exact bytes at zero, one, block-boundary and 800-sample extents, including cancellation, subnormals, signed zero and partitioned continuation.
ordered_rgba_has_a_nonassociative_negative_control|Would regrouping the sample sum be caught?|Use a cancellation example where a batch subtotal changes the answer; require the original ascending carry bytes.
ordered_rgba_refuses_invalid_extents_types_and_nonfinite_values|Do malformed or overflowing calls remain refused?|Check shape, dtype, 800-sample cap, nonfinite inputs and finite-input overflow without changing input storage.
ordered_rgba_old_provider_refuses_additive_operation|Can an older CUDA provider safely reject the new operation?|In an explicitly configured fresh process, require an unsupported-operation error before absent provider fields are accessed.
""")

_add('python/test_principled_forward_layout','Constant forward packing only. Original four-point obligations, native cohort limits and canonical backward group order remain unchanged.', """
constant_forward_avoids_group_slicing_and_preserves_ordered_vjp|Can constant forward cohorts avoid constructing temporary four-point lists?|Require no list-construction calls during forward, require the original group replay in backward, and compare all value and first-gradient bytes against the ordered four-point reference.
varying_fields_keep_general_group_layout|Do per-point material fields keep their existing path?|Require the general group constructor for varying albedo and compare output against the original four-point material calls.
""")

_add('python/test_opaque_face_selection','Forward-only conservative closed-window selection; all full input domain checks and dense differentiated quadrature remain required.', """
forward_native_quadrature_gets_only_window_faces_and_dense_backward|Does native forward execution receive only faces whose bounding boxes meet the window?|Observe ordinary and continued quadrature on an original crowded pixel; require at most 32 faces forward and preserve dense differentiated preparation.
unselected_material_cut_nonfinite_still_refuses|Can a bad off-window material cut escape validation?|Inject a nonfinite cut on a kept but off-window face and require rejection before accepting any image.
unreferenced_out_of_range_projected_vertex_still_refuses|Are all projected coordinates still checked?|Inject an out-of-range projected vertex and require the existing bounded-coordinate refusal even if selection could omit its faces.
""")

_add('python/test_opaque_horizon_partition','Each pixel must use its own generated horizon layout, independent of its neighbours or tile cap.', """
horizon_window_equals_canonical_pixels|Can neighbouring shading boundaries change a pixel when requests are regrouped?|Compare original-mesh windows at tile caps2,7,16 against every canonical single-pixel colour and coverage byte; a neighbouring horizon must not select a different integration rule.
""")

_add('python/test_opaque_host_requests','Host requests fit existing256-pixel storage blocks while internal tiles stay at most16; canonical replay and all resource limits remain.', """
host_request_profile_preserves_native_cap_and_block_storage|Do larger host requests remain bounded?|Observe a259-pixel row split into256 and3 with explicit16-pixel native hints and unchanged storage blocks.
host_request_profile_preserves_actual_pixels_and_ordered_replay|Does grouping change images or gradients?|Compare actual stored pixels,cotangents and canonical ordered gradients against the legacy driver; reject a corrupt request identity.
host_request_invalid_bounds_refuse_before_storage|Can an invalid host request allocate output storage?|Reject booleans,zero,negative,floating and257-pixel limits before creating a frame.
host_request_admission_preserves_full_anchor_identity|Can the new driver borrow incompatible timing evidence?|Require explicit bounded native/request caps and reject legacy-profile anchors for a larger frame.
""")

_add('python/test_opaque_cuda_preparation','Parallel domain checks and exact output padding only; ordered geometry and floating-point accumulation remain unchanged.', """
native_domain_checks_complete_input_tails|Are late invalid elements still rejected?|Place nonfinite/out-of-range vertices,bad face indices,cuts and cotangents at array tails; require refusal on ordinary and continued native CUDA paths.
native_empty_outputs_and_large_adjoint_padding_are_exact|Does parallel initialization preserve all padding bytes?|Require exact negative-one owners and positive-zero samples and large adjoint arrays for off-screen geometry on both native paths.
""")

_add('python/test_opaque_shared_horizons','Horizon coefficients are call-local forward data; exact crossing masks keep each pixel layout independent and backward remains fresh.', """
request_prepares_horizons_once_and_releases_them|Can a public row avoid rebuilding horizon coefficients for every internal tile?|Require one preparation instead of46,exact saved canonical original-row bytes,and release of all prepared tensors even while output autograd context survives.
active_zero_coefficients_keep_their_horizon_column|Can rounded-zero coefficients silently remove an active boundary column?|Use explicit crossing masks with zero coefficient arrays and require the correct linear columns and curved presence for each selection.
union_cut_capacity_falls_back_without_expanding_limits|Can a whole-request cut union refuse even though individual pixels fit?|Force union-capacity refusal,require one preparation attempt and exact canonical pixel fallback with no limit increase.
""")

_add('python/test_opaque_tile_cohort','Tile area must not silently sacrifice admitted material batching; all memory charges and exact pixel arithmetic remain in force.', """
original_row_keeps_grouped_shader_and_exact_pixels|Does a larger tile accidentally force thousands of tiny shader calls?|Render an original-mesh row,refuse unnecessary single-point material evaluation,and require exact saved canonical colour and mask bytes.
small_budget_single_point_plan_remains_available|Does preserving batching prohibit an already necessary small-budget fallback?|Supply a one-point baseline plan and verify that the public window still admits it and grows fitting tiles.
""")

_add('python/test_opaque_material_curve_reuse','Prepared scalar material curves belong to one bounded opaque forward request; backward still recomputes them and keeps its original order.', """
original_request_shares_curve_releases_it_and_refreshes_next_call|Can a row share constant material setup without stale values or retained graphs?|Require one preparation,exact original saved pixels,released curve tensors while the output context survives,and fresh preparation after changing roughness on the next call.
shared_curve_refuses_changed_resource_or_roughness|Can a reused curve silently survive modified inputs?|Change roughness and a resource tensor version separately and require explicit refusal from the forward guard.
""")

_add('python/test_opaque_forward_restore','Already planned forward requests reuse borrowed camera/scene bindings only within their version-guarded lifetime. Differentiated replay and each new request reconstruct and validate again.', """
forward_restores_camera_once_with_exact_original_row|Does every internal tile need to reconstruct the same camera?|Count one snapshot camera restoration for an original row and require exact saved colour and mask bytes.
preplanned_window_omits_unused_sample_camera|Is a planning camera still constructed after the workload was admitted?|Make unused800-sample camera construction fail and require the public preplanned window to render normally.
next_request_revalidates_mutated_camera|Can forward reuse conceal a camera changed between calls?|Render a valid request,make the camera pose singular,and require the next request to reject it.
""")

_add('python/test_opaque_empty_shading','An entirely uncovered forward window can avoid interpolation. Native coverage validation, every sample weight, reductions and differentiated replay remain unchanged.', """
uncovered_forward_omits_interpolation_with_exact_row|Does a blank window still interpolate geometry?|Count zero interpolation calls after native coverage returns no owners, while retaining exact positive-zero output bytes.
uncovered_background_retains_exact_signed_bytes|Does the shortcut alter background composition?|Compare signed and nonzero background bytes against the unchanged compositing expression.
uncovered_invalid_binding_still_rejected|Can a blank window hide invalid material inputs?|Require a nonfinite material binding to be rejected even outside the mesh.
uncovered_backward_retains_fresh_interpolation|Does this shortcut bypass differentiation?|Require differentiated replay to execute the original interpolation path and preserve the zero geometry gradient.
""")

_add('python/test_native_unit_newton','The native normalization primitive performs the same six rounded Newton steps. It is not a different square-root approximation, and the independent [1,3] proof and autograd formula remain in force.', """
unit_newton_matches_each_rounded_step|Does native fusion preserve the six-step arithmetic?|Compare exact eager-reference bytes for empty, singleton and larger bounded arrays on the selected device.
unit_newton_keeps_endpoint_neighbours_and_first_vjp|Are boundary-adjacent values and gradients unchanged?|Compare values and first gradients at normalization endpoints and their immediate floating-point neighbours.
unit_newton_refuses_invalid_inputs|Are malformed layouts and domains rejected?|Reject wrong dtype, rank, striding, oversized arrays and nonfinite or out-of-domain values.
unit_newton_raw_failure_atomicity_aliases_workspace_and_stream|Does the C interface honor caller-owned storage and streams?|Poison outputs, reject invalid domains, aliases, short scratch and context fields before writes, then compare a successful call on the caller CUDA stream.
unit_newton_old_provider_refuses_additive_operation|Does an older CUDA provider fail safely?|An isolated preserved-provider process must refuse the appended operation before input access.
""")

_add('python/test_native_unit_vectors','Complete forward normalization uses the same scaling, three-term sum, six Newton steps and division. Differentiated execution retains its original graph.', """
unit_vectors_preserve_extremes_signed_zero_and_rounding|Does fusion preserve extreme finite vectors and signed zeros?|Compare exact eager-reference bytes for tiny subnormals, huge finite vectors and bounded batch lengths.
unit_vectors_forward_uses_fused_path_with_exact_bytes|Do shading paths use the fused operation only in forward mode?|Make per-step square-root dispatch fail and require exact outputs from both forward normalization helpers.
unit_vectors_differentiated_path_remains_original|Is the differentiated normalization graph unchanged?|Forbid the forward-only native operation during differentiation and compare exact qualified first gradients.
unit_vectors_refuse_bad_shapes_domains_and_types|Are invalid vectors and layouts refused?|Reject zero or nonfinite vectors, wrong rank/dtype/stride and counts above the finite cap.
unit_vectors_raw_failure_atomicity_aliases_workspace_and_stream|Does the vector C interface preserve caller buffers on failure?|Verify poisoned-output preservation, aliases, short scratch, context fields and exact execution on the caller CUDA stream.
unit_vectors_old_provider_refuses_additive_operation|Does an older provider refuse the appended vector operation?|Use an isolated preserved-provider process and require explicit refusal.
""")

_add('python/test_deferred_stream_shared_curve','The fixed streamed albedo gradient shares one material table while retaining every native Jacobian column and the original accumulation order.', """
shared_stream_table_keeps_dense_gradient_bytes_and_full_jacobian|Does table sharing preserve the exact gradient?|Compare each original group with the dense path for positive, negative and tiny nonzero cotangents; require all sixteen derivative columns and forbid dense copies on the shared path.
shared_stream_reuses_table_only_inside_explicit_invocation|Does reuse stop at the invocation boundary?|Check one preparation for two calls sharing an explicit local state and a fresh preparation for a new state.
""")

_add('python/test_opaque_curve_only_windows',"A curved horizon alone does not change another pixel's linear cut count. Native per-pixel quadrature can therefore remain batched when exact pixel and gradient checks pass.", """
curve_only_forward_window_uses_native_batch_and_exact_pixels|Can curve-only windows stay grouped without changing pixels?|Observe native multi-pixel dispatch and compare every colour and mask byte with canonical one-pixel rendering.
curve_only_window_retains_canonical_normal_gradient|Does the faster forward preserve moving-normal gradients?|Compare exact pixels and canonical replay gradients at tile caps one and two.
zero_curve_rows_do_not_change_native_material_sample_bytes|Can an absent curve encoded as zeros alter sampling?|Compare every native owner and sample byte with and without zero conic rows, including an empty neighbour pixel.
""")

_add('python/test_deferred_stream_group_carry','Independent group reductions share dispatch while each group keeps its original tree. The native carry adds every group sequentially, without a combined subtotal.', """
stream_group_reductions_batch_independent_rows_without_padding|Can independent point reductions share dispatch?|Count reduction calls for all short leading-group sizes and require one batched tree for the remaining groups without padding.
stream_carry_preserves_every_addition_and_signed_tiny_terms|Does fused carry preserve cancellation and tiny contributions?|Compare every output byte with sequential additions for empty, single and full cohorts, signed zeros and very small nonzero terms.
stream_carry_refuses_nonfinite_parts_and_excess_groups|Are invalid gradient cohorts refused?|Reject nonfinite contributions and more than the existing thirty-two groups.
""")

_add('python/test_deferred_stream_light_cache','Reuse is limited to one guarded render invocation and eight light samples; complete scale admission remains separate.', """
stream_directional_packet_is_prepared_once_for_different_group_sizes|Can fixed light preparation be reused without changing gradients?|Compare independent original-autograd gradients for different cohort sizes and require exactly one single-point preparation.
stream_light_cache_cannot_escape_an_invocation|Can a previous render leak stale lighting into the next?|Change light direction between fresh invocations and compare every gradient byte with the independent uncached calculation.
stream_light_cache_keeps_larger_packets_on_uncached_path|Can light reuse silently exceed its reserved storage?|Use nine samples and require the original preparation path on each call, with exact gradients.
""")
_add('python/test_deferred_intermediate_admission','A complete intermediate frame uses the existing conservative admission formula; it is not a startup exception or an acceptance-matrix replacement.', """
intermediate_frame_uses_both_complete_anchors_and_unchanged_margin|Does an intermediate calibration weaken time admission?|Require both complete anchors, the slower pixel rate, twofold margin and thirty-second reserve for 768x768.
intermediate_frame_cannot_be_used_as_a_partial_or_mismatched_anchor|Can a stopped or changed execution become timing evidence?|Reject partial execution, failed status and mismatched driver identities.
""")

_add('python/test_native_delta_brdf','A bounded native CPU/CUDA lighting value and BRDF-only first VJP preserves the existing eager arithmetic. Other material and geometry derivatives remain separate.', """
delta_brdf_native_preserves_values_and_first_vjp|Can fused lighting change images or fitting directions?|Compare every value and BRDF-gradient byte with independent eager lighting and autograd across sample counts, full capacity and empty axes.
delta_brdf_native_retains_subnormal_and_signed_zero_terms|Are tiny values and signed zeros preserved?|Exercise subnormal cotangents, negative zero and the exact reverse reduction tree without tolerances or cutoffs.
delta_brdf_native_refuses_invalid_inputs|Does the new operation bypass the original input contract?|Reject nonfinite or negative values, zero vectors, invalid shape, dtype, stride and over-capacity layouts.
delta_brdf_raw_atomic_domains_aliases_workspace_and_stream|Can a failed call damage caller buffers or ignore stream ownership?|Use the installed C ABI on CPU and a caller CUDA stream; require unchanged sentinels on domain, alias, scratch and reserved-field failures.
delta_brdf_old_provider_refuses_additive_operation|Will an older CUDA provider fail safely?|Load a preserved provider in a fresh process and require explicit unsupported-operation refusal.
""")
_add('python/test_deferred_stream_delta_brdf','The fixed albedo helper uses native lighting only where its historical accumulation bytes are preserved. Multi-sample zero cotangents retain the original calculation.', """
stream_delta_brdf_uses_native_checked_lighting|Does the qualified helper actually use fused lighting?|Observe one native call and forbid repeated eager integration while retaining all material gradient groups.
stream_delta_brdf_retains_zero_and_tiny_complete_group_partials|Does integration preserve the previous complete group gradients?|Compare every partial byte with the saved preceding helper for one, three and eight light samples and signed zero or tiny cotangents.
""")

_add('python/test_deferred_stream_record_views','Views live only inside their source window. The unfinished queue becomes independently owned before the next allocation, within the existing carry reserve.', """
stream_group_views_avoid_a_copy_for_every_four_points|Does group assembly avoid hundreds of tiny device copies?|Count four-point clones during a complete backward stream and require copying to be limited to the bounded remainder.
stream_remainders_own_storage_before_source_window_is_released|Can a later window corrupt unfinished gradient groups or retain old window memory?|Create the largest allowed remainder, detach it, overwrite the source and verify unchanged owned values and the explicit storage bound.
""")

_add('python/test_raster_cooperative','Conservative block rejection preserves the original scan, including uncertain cases. These fixtures qualify arithmetic and indexing; a full-frame timing gate remains separate.', """
cooperative_raster_preserves_faces_blocks_and_partial_lanes|Can shared triangle rejection change a pixel or mishandle padded GPU threads?|Compare every output byte with pre-change CPU/CUDA captures across 255/256/257/513 faces, depth ties, offset rectangles and partial blocks in both precisions; preserve the existing empty-CUDA-face refusal.
""")

_add('python/test_shading_validation_readback','All checks and first-error precedence remain mandatory. Only the number of CUDA readbacks and temporary index selections changes; no validation is cached or skipped.', """
validation_retains_first_nonfinite_field_and_type_precedence|Can batching validation hide an invalid field or change the reported error?|Check ordered nonfinite failures, metadata precedence, empty inputs, signed zero and tiny finite values on CPU and CUDA.
cuda_validation_uses_one_readback_for_all_finite_fields|Does validation avoid one synchronization per input field?|Count CUDA scalar/list transfers for six finite fields and require one combined readback.
light_packet_refuses_every_nonzero_invalid_placeholder|Can an invalid light sample carry an unnoticed contribution?|Check every RGB component of invalid direction and irradiance placeholders, retaining tiny nonzero rejection and signed-zero acceptance.
material_domain_flags_keep_ordered_errors_and_empty_geometry|Do empty geometry or multiple bad parameters bypass material domains?|Require the original albedo, F0 and roughness error order even with no sample points, then accept valid endpoint values.
""")

_add('python/test_deferred_stream_validation','The fixed one-light GPU profile checks every finite, domain and placeholder obligation together. Failed or unsupported inputs run the original validators, so the first exception remains unchanged. Extra temporary memory is subtracted from the unchanged64MiB window allowance.', """
combined_cohort_validation_preserves_original|Can combining GPU predicates hide bad data or change errors?|Compare against the original validators for every finite field, material domains, negative light, tiny invalid placeholders, metadata, empty layouts, signed values and simultaneous failures.
combined_cohort_checks_finite_fields_once|Does the repair avoid repeated finite scans without omitting fields?|Count one finite scan over all1541 original scalar values in a complete128-point single-light cohort.
""")

_add('python/test_raster_cooperative','Conservative rejection may remove only faces outside the complete pixel rectangle. Surviving faces keep their order and exact arithmetic; inactive lanes never touch outputs but still reach every barrier.', """
cooperative_raster_awkward_spatial_rectangles|Do spatial GPU blocks retain pixels at partial rows and columns?|Compare every output byte with frozen prior results for wide, narrow, odd and one-pixel layouts in both precisions and devices.
""")

_add('python/test_deferred_stream_native_validation','Every value is still checked on each native call. Resource-specific roughness bounds and packet eligibility stay in Python. If native evaluation fails, the original validators recover the exact first diagnostic before failure is propagated; partial values never escape.', """
native_cohort_validation_retains_prior_errors|Does delegating repeated checks to native code preserve input failures?|Compare exact exception types and messages against the frozen prior helper for nonfinite fields, zero directions, all material domains, mutated light packets and simultaneous failures.
native_cohort_does_not_duplicate_native_finite_scan|Does the fixed helper avoid a second scan of already native-checked fields?|Require the full32-group result while observing that the redundant1541-scalar Python finite scan is absent; existing native and full-gradient controls still apply.
""")

_add('python/test_opaque_bounds_empty','Exact face extrema may be shared only inside a guarded forward request. An empty integration shortcut requires positive-zero weight bits; negative zero, tiny contributions and nonfinite values keep the original reduction.', """
forward_window_face_bounds_once|Are face bounds computed once per guarded window?|Observe one preparation across many tiles and exact saved zero output; existing mutation and backward tests guard lifetime.
empty_sample_reduction_guard|Can empty samples skip work without dropping tiny contributions?|Require no reduction only for positive-zero weights; compare exact prior arithmetic for negative zero, the smallest subnormal and NaN.
""")

_add('python/test_native_opaque_samples','This bounded primal composition is being qualified before scene selection. It preserves perspective interpolation, native Principled shading, directional lighting and the800-slot adjacent-pair integral. Full differentiated replay remains a separate existing path.', """
native_original_sample_composition_exact|Does native composition preserve the original mesh integral?|Compare colour, coverage and every weighted continuation sample with the installed independently qualified composition.
native_sample_nonfinite_refuses|Can nonfinite input fields reach computation?|Corrupt each floating field and require native refusal, including unused geometry and final sample slots.
native_sample_domain_refuses|Are geometry indices and material domains still enforced?|Reject invalid owners, vertices, material parameters, widths and irradiance.
native_sample_pixel_partition_and_capacity|Does processing several pixels change their results?|Repeat one original sample packet at1,2and16pixels and require every byte to match its single-pixel result.
native_sample_invalid_layout_refuses|Are invalid layouts and over-capacity calls refused?|Test empty/17pixel inputs, wrong types/shapes/strides and zero lighting directions.
native_sample_raw_atomic_domains_workspace_aliases_stream|Does the C ABI preserve caller storage and stream ownership on failures?|Check sentinel outputs after domain, workspace, alias and reserved-field refusal, then exact successful results on an explicit CUDA stream.
""")

_add('python/test_opaque_native_selection','The renderer selects native forward composition only for its qualified constant-material single-light profile. These controls ensure selection, backward fallback, first-error agreement and explicit storage charging.', """
constant_forward_selects_native_and_backward_keeps_eager|Does production use the qualified forward path without changing differentiated replay?|Observe selection during forward and eager replay during backward on the original mesh, checking finite gradients.
unsupported_native_profile_returns_to_reference|Do unsupported materials and lighting retain their established behavior?|Check two-sided bindings, explicit additional options and multiple lights through the eager fallback with finite output.
native_invalid_material_preserves_reference_error|Can the new path hide an invalid material or change its diagnostic?|Compare exception type and text for invalid albedo, roughness and F0 with native selection disabled.
native_storage_is_charged_inside_existing_allowance|Are native composition buffers charged against the existing memory limit?|Inventory maximal scratch and both outputs, and require the separate one MiB reserve in the opaque planner.
""")

_add('python/test_native_opaque_samples','These controls qualify exact special-value preservation and safe provider negotiation, not whole-frame rendering or gradients.', """
native_empty_signed_zero_and_small_weights|Are empty-slot signed zeros and subnormal contributions preserved exactly?|Compare the complete weighted array and ordered reduction bytes for positive zero, negative zero and signed smallest subnormals.
native_sample_old_provider_query_refuses|Does an older CUDA provider fail safely before new operation dispatch?|Load the preserved provider in a fresh process and require workspace-query refusal without modifying result sentinels.
""")

_add('python/test_opaque_empty_windows','A forward window can skip quadrature only after full validation and closed bounds prove no face can contribute. This does not skip differentiated replay or change continuation receipts.', """
validated_empty_request_omits_quadrature|Can a wholly empty request avoid per-tile quadrature while preserving its exact image?|Observe no quadrature call over a blank original-mesh row and compare every zero/background byte, including signed zero and a subnormal background.
empty_request_still_validates_inputs|Does empty-window acceleration accidentally hide invalid scene data?|Reject nonfinite background, vertices and material plus invalid triangle indices before any empty result escapes.
""")

_add('python/test_opaque_empty_windows','The empty fast return remains inside the public input-version guard. This adversarial control changes a real scene input during the preflight.', """
empty_preflight_refuses_mutation_before_return|Does an empty result still refuse scene mutation before returning?|Modify albedo immediately after the empty preflight and require the same public-window mutation error before output escapes.
""")

_add('python/test_opaque_native_capacity','The window stays within the original one-pixel legacy storage reservation plus the previously charged native reserve. Unsupported multi-pixel native calls must split before eager sample tensors are allocated; differentiated plans are unchanged.', """
native_forward_admits_sixteen_inside_existing_reserve|Can native forward composition use its supported sixteen-pixel window within 64 MiB?|Observe actual admitted capacity on the original mesh and check the complete planner reserve, then require finite image output.
unsupported_native_never_allocates_multi_pixel_eager|Can native fallback accidentally allocate the larger legacy sample graph?|Force native refusal, prohibit any multi-pixel eager call, and compare every result byte against canonical single-pixel evaluation.
differentiated_plan_keeps_original_storage|Does the forward storage optimization silently change a gradient-bearing plan?|Inspect the admitted gradient-bearing workload and require its original backward storage profile without native-window replacement.
""")

_add('python/test_opaque_native_capacity','Unknown or insufficient storage evidence cannot authorize a larger native window. The existing one-pixel planner remains the fallback.', """
missing_native_inventory_keeps_legacy_plan|Can incomplete planner metadata accidentally enable native capacity?|Supply missing and insufficient buffer inventories and require a conservative refusal of the optimization.
""")

_add('python/test_opaque_cuda_pixel_lanes','Four independent CUDA forward pixels have disjoint scratch and outputs. Every per-pixel operation and work cap remains unchanged; backward is serial. Extra scratch is charged inside the64MiB limit, and one-MiB native caller storage retains the serial fallback.', """
tight_public_budget_retains_serial_forward_and_first_vjp|Keep the existing public tight-budget path usable when extra lane storage cannot fit.|Compare complete CPU/CUDA owners,samples and first VJPs under two MiB; CUDA must fall back to minimum scratch.
parallel_curved_pixels_and_ordered_first_vjp_match_cpu|Keep curved horizon samples and the existing serial gradient reduction unchanged.|Compare every owner,sample,vertex and conic-coefficient VJP byte for four curved pixels on CPU/CUDA.
successful_neighbour_lanes_cannot_erase_capacity_failure|Preserve failure when only one concurrent pixel exceeds the candidate cap.|Place33candidates in the first pixel beside empty pixels and require overflow,never a successful partial window.
adapter_charges_and_allocates_four_disjoint_slabs|Prove the adapter really allocates the bounded lane storage and rejects a tight public budget.|Measure actual CUDA allocation after warmup and exercise budget refusal.
parallel_forward_keeps_scalar_pixel_bytes_on_nondefault_stream|Preserve CPU bytes for rectangular windows and both supported floating dtypes.|Compare every owner and sample byte for2/16pixels and a4x4window at a fractional origin on a nondefault CUDA stream.
native_caller_owned_scratch_keeps_minimum_fallback_and_padding|Qualify minimum and expanded caller scratch, exact padding and pre-write domain refusal.|Run1/2/4slabs with three prior byte patterns; compare exact output bytes and reject a malformed tail face while output sentinels remain intact.
""")

_add('python/test_opaque_albedo_replay','A bounded first-gradient replay reuses fixed geometry only within one call. Every pixel adds to the incoming RGB-albedo carry in the canonical order; other derivative targets retain ordinary autograd.', """
replay_public_facade_and_partition_carries|Expose the qualified operation through the lazy facade and preserve call partition independence.|Compare a mixed original boundary rectangle against canonical individual pixels and split calls with signed-zero and subnormal cotangents and carries.
replay_matches_canonical_pixels_and_ordered_carry|Preserve exact images and gradients under bounded replay.|Compare occupied, empty and continued original pixels against individual public calls, including signed-zero and subnormal carry values; verify borrowed inputs stay unchanged.
replay_prepares_fixed_geometry_once_and_does_not_persist|Avoid repeated geometry setup without a persistent cache.|Count preparation across two explicit calls and require exactly one preparation per call.
replay_refuses_unsupported_targets_or_invalid_inputs|Keep the qualified derivative and resource boundary explicit.|Refuse geometry gradients, bad cotangent shapes, nonfinite carry and requests above256pixels.
replay_refuses_input_mutation_before_result|Never return a replay result after its borrowed inputs change.|Mutate the original albedo during the first pixel and require a guarded failure before return.
""")

_add('python/test_opaque_frame_albedo_replay','The host stores every frame pixel and cotangent and calls the explicit bounded albedo replay. Exact stored bytes and an ascending gradient carry remain mandatory; no chunk reduction or persistent library cache is introduced.', """
frame_albedo_replay_preserves_exact_carry_and_checks_pixels|Verify complete finite host replay uses the bounded entry without changing results.|Compare the former individual-pixel gradient to bounded row calls, then corrupt a returned pixel and require rejection.
frame_albedo_profile_refuses_wrong_targets_before_storage|Refuse invalid opt-in profiles before creating frame storage.|Require a boolean option and exactly one RGB-albedo derivative target; reject geometry and missing targets.
""")

_add('python/test_principled_albedo_layout','Constant RGB-albedo backward keeps the full native sixteen-column Jacobian and VJP checks. The repair removes temporary framework graphs while preserving adjacent-pair reductions within original groups and reverse-group carry order.', """
constant_albedo_layout_scalar_f0_and_full_jacobian|Retain scalar F0 support and full sixteen-column derivatives.|Compare the retained graph with the compact layout while checking the native Jacobian still has all16columns.
constant_albedo_backward_avoids_temporary_group_graphs|Remove repeated temporary graph construction from fixed-albedo backward.|Trace the old material graph entry and require no call for the qualified profile.
constant_albedo_backward_matches_retained_graphs|Preserve exact gradient bytes, including tails and tiny signed contributions.|Compare the optimized path with its retained graph fallback across1..800points,1..8lights, signed-zero and subnormal cotangents.
other_derivative_targets_retain_original_graph_path|Keep general differentiation outside the optimization's scope.|Differentiate normals and albedo together and require the prior graph path.
constant_albedo_backward_preserves_native_derivative_failure|Never bypass full native derivative validation.|Inject a native VJP failure after a valid forward and require backward to propagate it.
""")

_add('python/test_opaque_original_frame_admission','The original frame time policy retains its approved startup sizes, conservative forecast and all numerical/resource prerequisites.', """
albedo_replay_profile_keeps_bounds_and_cannot_borrow_old_anchors|Give the new bounded albedo replay its own verified execution identity.|Check unchanged request caps, explicit backward scope and rejection of old-profile anchors; matching complete anchors retain the same conservative forecast.
""")

_add('python/test_opaque_original_frame_probe','A complete original-frame result must bind its declared execution profile to the runtime receipt and validate every stored byte.', """
frame_probe_forwards_replay_mode_and_verifies_receipt|Ensure a trial actually uses its declared replay path.|Intercept the real execution boundary for both modes and require exact option forwarding, shared deadline and rejection of a mismatched ownership receipt.
""")

_add('python/test_opaque_replay_curve_scope','Only the explicit fixed-albedo replay owns this prepared curve. Its scope resets even after failure; native value/domain/full derivative checks remain per invocation, and no cache lives on an autograd context or survives a public call.', """
replay_restores_callers_autograd_thread_setting|Keep scoped replay callbacks on the caller thread without changing the caller's later behavior.|Exercise both previous threading settings and failures, verify single preparation on the caller thread and exact setting restoration.
replay_curve_tensors_are_released_at_public_return|Verify prepared curve storage does not escape the bounded call.|Track weak references to all owned prepared tensors and require their release while returned graph-free outputs remain live.
replay_prepares_one_curve_per_call_for_forward_and_backward|Remove repeated preparation inside a bounded replay without persistent state.|Count native material-curve preparation across two calls, each containing four occupied pixels and their backward work.
replay_curve_scope_is_cleared_after_exception|Ensure failed calls cannot leak prepared material state.|Inject a replay error, inspect scope cleanup, then require a new independent call to succeed.
replay_curve_reuse_refuses_roughness_mutation|Reject results after a supposedly fixed material changes.|Change roughness during first curve construction and require a guarded failure plus cleanup.
curve_scope_refuses_different_storage_and_general_gradients|Reuse only the same guarded material storage and supported derivative scope.|Allow detached aliases, reject copied albedo storage and normal gradients, and verify cleanup.
""")


_add('python/test_opaque_fixed_lighting','The explicit fixed-albedo replay can reuse existing native normalization and single-delta lighting. General derivatives retain the original path. Every chunk keeps native domain, finite value and VJP checks; no approximation or reordered pixel sum is introduced.', """
fixed_replay_normalization_is_native_and_exact|Remove repeated constant normalization dispatch.|Compare exact extreme, subnormal and signed-zero results while forbidding the old scalar dispatch.
fixed_replay_normalization_keeps_live_derivatives|Preserve derivatives of changing normals.|Require the original normalization path and exact gradient bytes even inside a replay scope.
fixed_replay_lighting_preserves_value_and_vjp|Preserve complete lighting and BRDF derivatives.|Compare eager and native chunks across empty, tail and800point cases with ordinary, signed-zero and subnormal cotangents.
fixed_replay_lighting_uses_native_composition|Exercise the scoped optimized lighting path.|Forbid eager multiplication and require valid forward and backward through native chunks.
fixed_replay_lighting_retains_unsupported_fallback|Keep unsupported derivative and sample profiles on their established path.|Differentiate normals, direction or incoming light and exercise multiple or invalid samples while forbidding the scoped native operation.
fixed_replay_lighting_propagates_native_failure|Never swallow a native forward or VJP error.|Inject failure at either boundary and require it to reach the caller.
fixed_replay_lighting_retains_packet_and_domain_checks|Retain malformed-packet and mathematical-domain rejection.|Check negative or nonfinite light/material data, invalid placeholders and zero directions/normals.
""")


_add('python/test_native_opaque_albedo','This bounded primitive composes existing interpolation, normalization, the full sixteen-input Principled Jacobian and lighting reverse factors. It retains the original covered-point order, four-point pairwise reductions and reverse group carry. Raw qualification does not authorize production selection or a full frame.', """
native_sample_albedo_exact_eager_order|Preserve the established sample colour, mask and albedo VJP exactly.|Compare retained eager composition across empty groups, group/cohort tails, signed zeros and subnormal cotangents.
native_sample_albedo_pixel_partition|Keep independent pixel lanes equivalent to individual calls.|Compare all outputs and derivatives for one,two and sixteen identical independent pixels.
native_sample_albedo_nonfinite_refusal|Reject nonfinite input tails and invalid indices before exposing output.|Corrupt each input in turn and require refusal.
native_sample_albedo_domain_refusal|Keep fixed raw layout and physical-domain guards.|Exercise over-capacity, wrong shape/stride, negative albedo and zero light/curve inputs.
native_albedo_raw_atomic_domains_workspace_aliases_stream|Validate the C ABI on caller-owned storage and streams.|Require unchanged outputs on invalid data, insufficient/aliased workspace and overlapping outputs; then verify exact valid results on a nondefault stream.
native_albedo_retains_tiny_weights|Do not discard zero or tiny quadrature contributions.|Compare the eager result with positive/negative zero and the smallest signed subnormal weights.
native_albedo_old_provider_query_refuses|Refuse the additive operation with an older CUDA provider.|A separate fresh-process control requires unsupported-query refusal without modifying size/alignment outputs.
""")


_add('python/test_opaque_native_albedo_replay','The explicit fixed-albedo replay may use the qualified native sample VJP for up to eight independent ordinary pixels at a time. It retains the existing ordered pixel carry, guarded call-local curve and every resource limit. Overflow continuation and general differentiation retain their established paths.', """
replay_uses_native_sample_vjp_for_ordinary_pixels|Connect the bounded native VJP to the actual public replay.|Forbid eager sample composition and trace one bounded native group for the four ordinary pixels while requiring finite output.
native_replay_failure_is_not_hidden|Preserve native error reporting.|Inject a native sample VJP failure and require the public replay to fail rather than silently accept fallback output.
general_window_differentiation_does_not_select_explicit_native_vjp|Keep general autograd outside the optimization.|Forbid the new primitive during an ordinary differentiable window and verify the established gradient remains finite.
""")

_add('python/test_opaque_albedo_groups','At most eight independent native pixel VJPs share dispatch. Each pixel keeps its exact arithmetic; gradients are added in ascending pixel order, never as a group subtotal. Existing memory and work caps remain.', """
eight_pixel_native_batches_and_tail|Use bounded groups and preserve a short tail.|Trace native dispatch sizes8,8,1 for seventeen ordinary pixels and require graph-free outputs.
grouped_replay_exact_canonical_carries|Retain exact image and gradient bytes.|Compare ordinary, row-boundary, empty and overflow regions with canonical individual pixels using signed zero, subnormal cotangents and a nonzero carry.
group_capacity_refusal_retries_only_canonical_pixels|A refused group does not lose pixels.|Inject group-capacity refusal and require all eight canonical pixel calls.
wide_unsupported_native_never_allocates_eager_group|Keep fallback storage bounded.|Refuse native groups and assert every eager sample call stays at one pixel while all pixels are replayed.
""")

_add('python/test_opaque_native_albedo_pixel','Fixed-albedo continuation computes each bounded sample batch and full checked derivative together. It preserves ordered RGBA samples, ascending gradient batches, all receipts and aggregate work even when an unsupported native profile falls back.', """
fixed_overflow_native_matches_exact_without_engine_replay|Eliminate repeated general backward work without changing results.|All three saved difficult pixels must match canonical image and signed/tiny gradient bytes with the general autograd engine forbidden during explicit replay.
native_continuation_failure_propagates|Do not conceal a failed native continuation.|An injected native first-VJP failure must escape the public replay.
unsupported_continuation_keeps_canonical_fallback_and_shared_work|Preserve complete fallback and its aggregate work counter.|Refuse the explicit native batch and require exact canonical output with one work counter across the attempt and fallback.
native_continuation_rejects_receipt_corruption|Keep continuation prefix and completeness safeguards.|Corrupt a native batch slot total and require failure before returning a complete pixel.
""")


_add('python/test_opaque_offline','Larger original images are qualified offline in bounded serial row ranges. Global cameras and full-frame cotangents remain fixed; exact incoming gradient bytes preserve pixel addition order. These checks establish the range harness, not large-frame accuracy or interactive speed.', """
offline_range_preserves_global_camera_tail_and_carry|Splitting a frame must preserve the complete image and gradient.|Compare two global row ranges including a short tail with the canonical complete small frame; require every stored pixel and final gradient byte to match.
offline_nonzero_carry_and_varying_cotangents|Do not round away small terms or replace the accumulated gradient with a subtotal.|Compare nonzero incoming carries, signed zero and subnormal cotangents against ordered individual pixel derivatives.
offline_rejects_missing_reordered_corrupt_and_changed_inputs|Incomplete or corrupted work cannot become a complete frame.|Refuse missing or duplicate ranges, a changed cotangent plan, changed camera inputs and corrupted pixel bytes.
offline_cancellation_exhaustion_and_pixel_mismatch_never_seal|A failed range must not be recorded as complete.|Inject cancellation, a consumed deadline and a recomputed pixel mismatch; require no sealed range receipt.
offline_policy_plan_caps|Keep the approved offline workload finite.|Check exact eight-range512 and sixty-four-range1080 coverage and reject altered extents or a range exceeding32768pixels.
offline_receipt_and_resource_negative_controls|Protect saved progress and original inputs.|Reject changed carry, predecessor, origin, camera identity, reversed ranges, aggregation cancellation, injected memory exhaustion and scene mutation.
offline_supervised_report_required|A saved artifact alone does not prove its worker passed.|Require matching supervised PID, successful exit, original index and bounded measured time, watchdog, host memory and GPU allocation.
""")


_add('python/test_flame_photometric','Group4 uses gffx plus PyTorch for a FLAME photometric-fitting experiment. These independent synthetic controls qualify the composition before model assets or a real photograph are used. The initial renderer is an explicitly labelled hard-raster, normalized-normal SH variant; no real FLAME fit, upstream byte parity or final showcase is claimed.', """
model_joint_regression_pose_correctives_bind_and_landmarks|Deformation must include shape-dependent joints and pose correctives.|Compare a two-joint model with closed independent geometry and embedded landmarks, preserving fixed tensors.
model_shape_and_joint_gradients_independent_and_two_scale|Shape and jaw gradients must match the actual deformation.|Compare independent analytic derivatives and central differences at two step sizes.
camera_matches_reference_scale_translation_flip_and_gradients|Camera signs and scale must agree with the declared image coordinates.|Compare the native transform with explicit scaled-orthographic equations and derivatives.
texture_basis_channels_and_coefficient_derivative|Texture coefficients must retain RGB meanings and gradients.|Compare the generated texels and coefficient adjoint with the explicit affine model.
sh_coefficient_conversion_independent_unit_normal_basis|Lighting coefficient order and normalization must be explicit.|Compare gffx SH output and coefficient gradients with the independently written upstream unit-normal basis.
losses_mask_denominator_landmark_norm_and_regularizers|Mask normalization must not silently change the objective.|Verify masked L1 over the full image, Euclidean landmark distances and shape/expression priors.
model_refuses_invalid_inputs|Invalid models must fail before fitting.|Reject nonfinite coefficients, wrong coefficient dimensions and cyclic joint parents.
image_matches_independent_affine_texture_and_camera_gradient|The image must agree with an independent pixel formula.|Compare affine texture pixels and camera derivatives through rasterization and texture sampling.
image_texture_light_geometry_vjp_two_scale|Image gradients must reach geometry, appearance and lighting.|Check every chosen direction against two finite-difference scales on stable raster support.
bounded_optimizer_recovers_independent_texture_target|A controlled fit must recover the known appearance.|Recover three texture offsets from an independent affine target in64steps, preserving fixed texels.
optimizer_refuses_cancel_exhaustion_and_nonfinite|A failed stage cannot silently continue updating parameters.|Reject cancellation, exhausted time, invalid iteration allowance and nonfinite loss before an update.
image_profile_refuses_invalid_extent_and_zero_area|Finite guards are enforced.|Refuse an oversized image and zero-area geometry with invalid shading normals.
five_joint_pose_layout_and_dynamic_contour_bins|Global and jaw pose must occupy the correct joints.|Check fixed neck/eyes and positive,negative and saturated yaw contour indices.
complete_parameter_chain_loss_derivatives_before_optimization|All six parameter groups must influence the complete fitting objective correctly.|Compare shape,expression,pose,camera,texture and lighting directional derivatives at two finite-difference scales.
complete_chain_fresh_step_decreases_fixed_objective|The connected model must support a fresh useful update.|Run eight bounded Adam steps on the fixed synthetic photometric/landmark objective and verify lower final loss.
""")

_add('python/test_sh_fixed_contraction','Spherical-harmonic illumination must preserve the declared CPU/CUDA arithmetic in both images and first gradients. Fixed products and reductions replace library-dependent matrix contraction; these controls do not imply whole-model fitting convergence.', """
sh_fixed_contraction_forward_and_first_vjp_bytes|Lighting output and its first derivatives must agree exactly on CPU and CUDA.|Compare identical nontrivial inputs and cotangents for singleton/shared/distinct batches and empty points using raw bytes.
sh_contraction_independent_sum_and_directional_derivative|Exact device agreement must also retain the right mathematical function.|Compare an independent high-accuracy nine-term sum and two-scale directional differences for normals and coefficients.
""")

_add('python/test_flame_photometric','Complete synthetic fitting-chain CPU/CUDA conformance is a separate gate from numerical accuracy. This fixture exercises six parameter groups with fixed observations; it does not prove real FLAME fitting or arbitrary dimensions.', """
complete_chain_exact_cpu_cuda_output_and_gradient_bytes|The connected example must preserve exact CPU/CUDA images and all six first gradients.|Compare raw bytes for colour,mask,depth,landmarks,vertices,albedo and the shape,expression,pose,camera,texture and lighting adjoints.
""")

_add('python/test_flame_assets','These data-conversion controls preserve selected FLAME arrays and reject unsupported serialized inputs under an external worker cap. They do not establish real fitting convergence, trusted annotations, or safety for arbitrary untrusted uploads.', """
restricted_legacy_numeric_roundtrip_and_hash_refusal|Use the exact selected numeric data.|Decode a synthetic array dictionary and reject a changed source hash.
restricted_legacy_refuses_arbitrary_callable|Serialized data must not invoke arbitrary functions.|An eval-bearing pickle must be refused by the allowlist before invocation.
geometry_conversion_explicit_expression_offset_and_tree|Select the intended shape and expression components.|Compare explicit source slices at shape0,1 and expression300,301 and preserve the rooted joint tree.
geometry_conversion_rejects_invalid_data|Invalid model data must stop conversion.|Reject out-of-range faces, unnormalized weights, invalid parents and nonfinite basis entries.
streamed_texture_selection_matches_independent_channels_and_nearest|Streaming must preserve selected texels and component meanings.|Compare with direct independent nearest-neighbour selection, BGR-to-RGB conversion and byte scaling on a small known texture.
streamed_texture_rejects_unsupported_or_incomplete_input|Do not silently reinterpret a texture archive.|Reject object or Fortran layouts, insufficient components and a truncated ZIP.
obj_preserves_face_corner_uv_order_and_rejects_quad|UV corners must follow their original face indices.|Check an explicitly permuted triangle and its v-down coordinates, then reject a quad.
landmark_tables_validate_shape_weights_and_face_indices|Landmark embeddings must refer to valid model triangles.|Validate all three table extents and reject an out-of-range face.
legacy_torch_embedding_decodes_as_arrays_without_torch_execution|Legacy tensor data must remain numeric arrays.|Decode known integer and float tensors through inert storage-envelope parsing and compare all values.
legacy_torch_storage_and_tensor_reject_invalid_inputs|Unsupported tensor envelopes must fail explicitly.|Reject truncation, nested callable globals, noncontiguous strides, offsets and gradient-bearing tensors.
full_landmarks_accept_only_explicit_singleton_batch|Remove only the documented redundant full-landmark batch axis.|Accept one batch of68 landmarks and reject an incompatible two-batch table.
landmark_partition_check_uses_stored_values_without_float32_sum_error|Validate stored barycentric weights without introducing float32 summation error.|Use a counterexample whose exact stored-value sum meets the unchanged1e-7 criterion, preserving every weight after float64 promotion.
""")

_add('python/test_flame_photometric','Fixed model contractions cover the actual5023-vertex regression and five-joint inverse-bind layouts. Exact arithmetic and independent formula checks are separate from full fitting acceptance.', """
fixed_model_matrix_products_and_vjps_real_extents|Model matrix products and their first gradients must agree exactly across devices.|Compare identical real-size matrix data, broadcast batches and adjoints byte-for-byte; also compare an independent high-accuracy scalar sum and ordinary differentiable matrix algebra.
""")

_add('python/test_portrait_review','This read-only review displays draft annotations, not a fitted image or an approval. Data and the unchanged photograph remain in internal dashboard storage.', """
portrait_review_returns_only_verified_data_and_draft_status|The review must show the selected image and explicitly remain a draft.|Verify the content-addressed image URL,68points and exact review-data hash; no acceptance is inferred.
portrait_review_refuses_invalid_source_or_coordinates|Malformed inputs must not become a plausible overlay.|Reject non-hash paths, nonfinite/out-of-image points, reversed crops, invalid polygons and PNG dimension mismatches.
portrait_review_refuses_changed_image_bytes|The overlay must not silently move to a changed photograph.|Alter saved PNG bytes and require content-hash refusal before serving review data.
""")
