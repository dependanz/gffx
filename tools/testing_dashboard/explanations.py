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
