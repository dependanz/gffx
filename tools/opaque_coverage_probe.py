"""Serial bounded oracle/coverage qualification; never substitutes a source overlay."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from tools import material_loss_cost as cost

WHEEL=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/scale50-20260919/wheels1/gffx-0.2.0.dev0-cp310-abi3-win_amd64.whl')
SHA='c4357c9746a9724ad6624c3392c3d6eb28ec3c4c278b8399415545389c64a5d2'
MODES={'original-acceptance':['tests/python/test_opaque_original_acceptance.py'],'opaque-cuda-original-acceptance':['tests/python/test_opaque_original_acceptance.py'],'original-radial-horizon':['tests/python/test_opaque_original_radial_horizon.py'],'original-radial-maps':['tests/python/test_opaque_original_radial_maps.py'],'original-radial':['tests/python/test_opaque_original_radial.py'],'radial-resample-controls':['tests/packaging/test_opaque_radial_resample.py'],'original-subpixel':['tests/python/test_opaque_original_subpixel.py'],'original-piece-error':['tests/python/test_opaque_original_piece_error.py'],'offline-direction-tail':['tests/packaging/test_opaque_direction_preflight.py'],'prechecked-regions':['tests/packaging/test_opaque_prechecked_regions.py'],'direction-precheck':['tests/packaging/test_opaque_direction_precheck.py'],'complementary-regions':['tests/packaging/test_opaque_complementary_regions.py'],'offline-value-tail':['tests/packaging/test_opaque_value_preflight.py'],'value-tail':['tests/packaging/test_opaque_value_tail.py'],'value-cost':['tests/packaging/test_opaque_value_cost.py'],'value-regions':['tests/packaging/test_opaque_value_regions.py'],'centered-value':['tests/packaging/test_opaque_centered_value.py'],'turn-diagnostic':['tests/packaging/test_opaque_turn_diagnostic.py'],'offline-centered-tail':['tests/packaging/test_opaque_centered_preflight.py'],'tail-boundary':['tests/packaging/test_opaque_tail_boundary.py'],'centered-partial':['tests/packaging/test_opaque_centered_gradient.py::test_centered_gradient_refines_only_unit_auxiliary_partial'],'centered-regions':['tests/packaging/test_opaque_centered_regions.py'],'117-regression':['tests/packaging/test_opaque_boundary_material.py','tests/packaging/test_opaque_high_material.py','tests/packaging/test_opaque_high_transport.py'],'centered-gradient':['tests/packaging/test_opaque_centered_gradient.py'],'tail-monotonicity':['tests/packaging/test_opaque_tail_monotonicity.py'],'offline-tail-preflight':['tests/packaging/test_opaque_offline_tail_preflight.py'],'116-profile-regression':['tests/packaging/test_opaque_offline_profile.py','tests/packaging/test_opaque_twelfth_boundary.py::test_tail_preflight_requires_explicit_offline_profile'],'tail-profile':['tests/packaging/test_opaque_twelfth_boundary.py::test_tail_preflight_requires_explicit_offline_profile'],'twelfth-boundary':['tests/packaging/test_opaque_twelfth_boundary.py'],'116-regression':['tests/packaging/test_opaque_late_boundary.py','tests/packaging/test_opaque_selective_boundary.py'],'twelfth-leaf':['tests/packaging/test_opaque_twelfth_leaf_diagnostic.py'],'selective-preflight':['tests/packaging/test_opaque_selective_preflight.py'],'selective-boundary':['tests/packaging/test_opaque_selective_boundary.py'],'decimal-cost':['tests/packaging/test_opaque_decimal_cost.py'],'decimal-diagnostic':['tests/packaging/test_opaque_decimal_diagnostic.py'],'decimal-material':['tests/packaging/test_opaque_decimal_material.py'],'decimal-native':['tests/packaging/test_opaque_decimal_native.py'],'native-diagnostic':['tests/packaging/test_opaque_native_diagnostic.py'],'native-material':['tests/packaging/test_opaque_native_material.py'],'native-convolution':['tests/packaging/test_opaque_native_convolution.py'],'curve-prefix':['tests/packaging/test_opaque_curve_prefix.py'],'algebraic-diagnostic':['tests/packaging/test_opaque_algebraic_diagnostic.py'],'algebraic-binding':['tests/packaging/test_opaque_algebraic_binding.py'],'112-regression':['tests/packaging/test_opaque_boundary_taylor.py','tests/packaging/test_opaque_boundary_material.py','tests/packaging/test_opaque_radial_reuse.py','tests/packaging/test_opaque_material_inputs.py'],'algebraic-material':['tests/packaging/test_opaque_algebraic_material.py'],'algebraic-boundary':['tests/packaging/test_opaque_algebraic_boundary.py'],'arb-series-diagnostic':['tests/packaging/test_opaque_series_diagnostic.py'],'arb-series-material':['tests/packaging/test_opaque_series_material.py'],'arb-persistent-series':['tests/packaging/test_opaque_persistent_series.py'],'direct-preflight':['tests/packaging/test_opaque_direct_preflight.py'],'direct-aggregate-cost':['tests/packaging/test_opaque_direct_aggregate_cost.py'],'111-regression':['tests/packaging/test_opaque_residual_range.py','tests/packaging/test_opaque_aggregate_peano.py','tests/packaging/test_opaque_aggregate_transport.py'],'direct-aggregate':['tests/packaging/test_opaque_direct_aggregate.py'],'seeded-preflight':['tests/packaging/test_opaque_seeded_preflight.py'],'seeded-cover':['tests/packaging/test_opaque_seeded_cover.py'],'boundary-jump-regions':['tests/packaging/test_opaque_boundary_jump_regions.py'],'boundary-jump-diagnostic':['tests/packaging/test_opaque_boundary_jump_diagnostic.py'],'boundary-jump-localization':['tests/packaging/test_opaque_boundary_jump_localization.py'],'boundary-order3-diagnostic':['tests/packaging/test_opaque_boundary_order3_diagnostic.py'],'boundary-terms-diagnostic':['tests/packaging/test_opaque_boundary_terms_diagnostic.py'],'boundary-local-diagnostic':['tests/packaging/test_opaque_boundary_local_diagnostic.py'],'boundary-local-base':['tests/packaging/test_opaque_boundary_local_base.py'],'boundary-order4-diagnostic':['tests/packaging/test_opaque_boundary_order4_diagnostic.py'],'110-boundary-regression':['tests/packaging/test_opaque_boundary_taylor.py','tests/packaging/test_opaque_boundary_material.py'],'boundary-order4':['tests/packaging/test_opaque_boundary_order4.py'],'signed-prefix-diagnostic':['tests/packaging/test_opaque_signed_prefix_diagnostic.py'],'radial-reuse':['tests/packaging/test_opaque_radial_reuse.py'],'arb-persistent-diagnostic':['tests/packaging/test_opaque_persistent_diagnostic.py'],'arb-persistent-material':['tests/packaging/test_opaque_persistent_material.py'],'arb-intervals':['tests/packaging/test_opaque_arb_intervals.py'],'kernel-boundary-profile':['tests/packaging/test_opaque_kernel_boundary_profile.py'],'kernel-power-reuse':['tests/packaging/test_opaque_kernel_power_reuse.py'],'109-kernel-controls':['tests/packaging/test_opaque_high_peano.py','tests/packaging/test_opaque_high_material.py'],'binary-diagnostic':['tests/packaging/test_opaque_binary_diagnostic.py'],'binary-material':['tests/packaging/test_opaque_binary_material.py'],'binary-derivatives':['tests/packaging/test_opaque_binary_derivatives.py'],'map-aggregate-diagnostic':['tests/packaging/test_opaque_map_aggregate_diagnostic.py'],'sharp-diagnostic':['tests/packaging/test_opaque_sharp_diagnostic.py'],'sharp-range':['tests/packaging/test_opaque_sharp_range.py'],'affine-feasibility':['tests/packaging/test_opaque_affine_feasibility.py'],'108-regression':['tests/packaging/test_opaque_input_reuse.py','tests/packaging/test_opaque_horizon_factor.py','tests/packaging/test_opaque_peano.py','tests/packaging/test_opaque_branch_partition.py'],'material-input-cost':['tests/packaging/test_opaque_material_input_cost.py'],'material-inputs':['tests/packaging/test_opaque_material_inputs.py'],'primitive-hybrid':['tests/packaging/test_opaque_primitive_hybrid.py'],'primitive-diagnostic':['tests/packaging/test_opaque_primitive_diagnostic.py'],'primitive-rectangle':['tests/packaging/test_opaque_primitive_rectangle.py'],'primitive-parts':['tests/packaging/test_opaque_primitive_parts.py'],'angular-primitives':['tests/packaging/test_opaque_angular_primitives.py'],'gauss8-preflight':['tests/packaging/test_opaque_gauss8_preflight.py'],'gauss8-correction':['tests/packaging/test_opaque_gauss8_correction.py'],'106-regression':['tests/packaging/test_opaque_square_reference.py','tests/packaging/test_opaque_refinement_guidance.py::test_guided_nonlinear_integral_and_tangent_still_match_closed_form','tests/packaging/test_opaque_refinement_guidance.py::test_zero_or_tiny_guide_never_admits_unresolved_bound','tests/packaging/test_opaque_refinement_guidance.py::test_invalid_guidance_falls_back_but_stop_propagates','tests/packaging/test_opaque_refinement_guidance.py::test_guidance_does_not_change_exact_one_cell_evidence','tests/packaging/test_opaque_refinement_guidance.py::test_original_checker_supplies_separate_refinement_field','tests/packaging/test_opaque_conserved_allocation.py','tests/packaging/test_opaque_tensor_order.py::test_original_profile_uses_qualified_fourth_piece_order'],'gauss8-square':['tests/packaging/test_opaque_gauss8_square.py'],'gauss8-material':['tests/packaging/test_opaque_gauss8_material.py'],'gauss8-diagnostic':['tests/packaging/test_opaque_gauss8_diagnostic.py'],'gauss8':['tests/packaging/test_opaque_gauss8.py'],'105-regression':['tests/packaging/test_opaque_high_material.py','tests/packaging/test_opaque_high_transport.py'],'arb-material':['tests/packaging/test_opaque_arb_material.py'],'arb-diagnostic':['tests/packaging/test_opaque_arb_diagnostic.py'],'arb-jets':['tests/packaging/test_opaque_arb_jets.py'],'boundary-coefficient-diagnostic':['tests/packaging/test_opaque_boundary_coefficient_diagnostic.py'],'boundary-material':['tests/packaging/test_opaque_boundary_material.py'],'boundary-taylor':['tests/packaging/test_opaque_boundary_taylor.py'],'104-regression':['tests/packaging/test_opaque_high_peano.py','tests/packaging/test_opaque_high_material.py','tests/packaging/test_opaque_grouped_peano.py','tests/packaging/test_opaque_transverse_peano.py','tests/packaging/test_opaque_residual_range.py'],'transport-diagnostic':['tests/packaging/test_opaque_transport_diagnostic.py'],'high-transport':['tests/packaging/test_opaque_high_transport.py'],'term-diagnostic':['tests/packaging/test_opaque_term_diagnostic.py'],'order-diagnostic':['tests/packaging/test_opaque_order_diagnostic.py'],'high-diagnostic':['tests/packaging/test_opaque_high_diagnostic.py'],'high-material':['tests/packaging/test_opaque_high_material.py'],'high-peano':['tests/packaging/test_opaque_high_peano.py'],'grouped-diagnostic':['tests/packaging/test_opaque_grouped_diagnostic.py'],'grouped-peano':['tests/packaging/test_opaque_grouped_peano.py'],'aggregate-transport':['tests/packaging/test_opaque_aggregate_transport.py'],'aggregate-diagnostic':['tests/packaging/test_opaque_aggregate_diagnostic.py'],'aggregate-cost':['tests/packaging/test_opaque_aggregate_cost.py'],'aggregate-peano':['tests/packaging/test_opaque_aggregate_peano.py'],'residual-range-cost':['tests/packaging/test_opaque_residual_range_cost.py'],'residual-range':['tests/packaging/test_opaque_residual_range.py'],'monotone-bounds':['tests/packaging/test_opaque_monotone_bounds.py'],'domain-direction3':['tests/packaging/test_opaque_domain_direction3.py'],'horizon-power':['tests/packaging/test_opaque_horizon_power.py::test_flattened_map_has_independent_closed_area_and_boundary_motion','tests/packaging/test_opaque_horizon_power.py::test_flattened_map_preserves_point_coordinate_jacobian_and_factored_cosine','tests/packaging/test_opaque_horizon_power.py::test_unqualified_powers_are_refused'], 'ninth-preflight':['tests/packaging/test_opaque_horizon_power.py::test_original_ninth_piece_bounded_preflight'],'domain-direction2':['tests/packaging/test_opaque_domain_direction2.py'],'domain-split':['tests/packaging/test_opaque_domain_split.py'],'horizon-factor':['tests/packaging/test_opaque_horizon_factor.py'],'domain-direction':['tests/packaging/test_opaque_domain_direction.py'],'allocation-composition':['tests/packaging/test_opaque_conserved_allocation.py::test_nonzero_width_certificates_compose_at_unchanged_final_bound'],'conserved-allocation':['tests/packaging/test_opaque_conserved_allocation.py'],'kernel-power-cost':['tests/packaging/test_opaque_kernel_power_cost.py'],'kernel-power':['tests/packaging/test_opaque_kernel_power_cache.py','tests/packaging/test_opaque_direct_kernel.py','tests/packaging/test_opaque_kernel_mass.py'],'lookup-audit':['tests/packaging/test_opaque_timing_variance.py::test_completed_lookup_diagnostic_is_auditable_without_rerun'],'lookup-cost':['tests/packaging/test_opaque_timing_variance.py::test_lookup_same_regions_with_remaining_budget'],'knot-lookup':['tests/packaging/test_opaque_knot_lookup.py'],'timing-variance':['tests/packaging/test_opaque_timing_variance.py::test_second_piece_same_regions_old_new_timing'],'certified-priority':['tests/packaging/test_opaque_refined_roots.py::test_original_piece_guidance_and_optional_root_refinement_binding','tests/packaging/test_opaque_refined_roots.py::test_certified_width_priority_keeps_closed_integral_and_refuses_cell_exhaustion'],'refined-roots-cost':['tests/packaging/test_opaque_refined_roots_cost.py'],'refined-roots':['tests/packaging/test_opaque_refined_roots.py'],'mass-no-duplicate':['tests/packaging/test_opaque_kernel_mass.py::test_mass_does_not_rebuild_the_same_segment_polynomials_for_rectangle'],'kernel-cache-controls':['tests/packaging/test_opaque_conditioned_center.py::test_center_and_kernel_reuse_preserve_exact_acceptance_and_charge_work'],'kernel-mass-cost':['tests/packaging/test_opaque_direct_kernel_cost.py::test_mass_same_regions_with_remaining_diagnostic_allowance'],'kernel-cost-audit':['tests/packaging/test_opaque_direct_kernel_cost.py::test_saved_direct_kernel_comparison_uses_outward_margin'],'kernel-mass':['tests/packaging/test_opaque_kernel_mass.py'],'direct-kernel-cost':['tests/packaging/test_opaque_direct_kernel_cost.py'],'direct-kernel':['tests/packaging/test_opaque_direct_kernel.py'],'kernel-regression':['tests/packaging/test_opaque_peano.py','tests/packaging/test_opaque_requested_kernel.py','tests/packaging/test_opaque_signed_peano.py','tests/packaging/test_opaque_transverse_peano.py','tests/packaging/test_opaque_localized_peano.py','tests/packaging/test_opaque_segmented_remainder.py','tests/packaging/test_opaque_signed_partition.py','tests/packaging/test_opaque_smooth_peano.py'],'adaptive-acceptance-guard':['tests/packaging/test_opaque_adaptive_order.py::test_small_selection_score_never_replaces_full_conditioned_error'],'adaptive-order':['tests/packaging/test_opaque_adaptive_order.py::test_small_selection_score_never_replaces_full_conditioned_error','tests/packaging/test_opaque_adaptive_order.py::test_adaptive_material_keeps_independent_analytical_integrals', 'tests/packaging/test_opaque_adaptive_order.py::test_selector_charges_both_candidates_reuses_chosen_rows_and_contains_exact_polynomial', 'tests/packaging/test_opaque_adaptive_order.py::test_tie_keeps_preferred_axis_and_failure_of_alternate_keeps_valid_bound', 'tests/packaging/test_opaque_adaptive_order.py::test_cancellation_and_two_unavailable_orders_cannot_be_accepted', 'tests/packaging/test_opaque_adaptive_order.py::test_invalid_selector_is_refused_before_callbacks', 'tests/packaging/test_opaque_adaptive_order.py::test_only_fourth_original_piece_enables_adaptive_order'],'adaptive-order-diagnostic':['tests/packaging/test_opaque_adaptive_order.py::test_same_saved_regions_adaptive_order_bounds_and_cost'],'d351-diagnostic':['tests/packaging/test_opaque_d351_diagnostic.py'],'fused-products':['tests/packaging/test_opaque_fused_products.py'],'zero-material-work':['tests/packaging/test_opaque_zero_material_work.py'],'owned-diagnostic':['tests/packaging/test_opaque_owned_jets.py::test_saved_regions_retain_exact_bounds_with_owned_arithmetic'],'owned-jets':['tests/packaging/test_opaque_owned_jets.py::test_arithmetic_does_not_recopy_owned_coefficient_dictionary','tests/packaging/test_opaque_owned_jets.py::test_result_mutation_cannot_change_inputs','tests/packaging/test_opaque_owned_jets.py::test_external_coefficient_dictionary_still_copies','tests/packaging/test_opaque_owned_jets.py::test_addition_uses_one_context_and_directed_endpoints','tests/packaging/test_opaque_owned_jets.py::test_decimal_endpoint_construction_avoids_conversion','tests/packaging/test_opaque_owned_jets.py::test_decimal_fast_path_retains_invalid_endpoint_refusal','tests/packaging/test_opaque_interval_work.py','tests/packaging/test_opaque_denominator_reuse.py'],'d347-split':['tests/packaging/test_opaque_d347_split.py'],'tensor-order-diagnostic':['tests/packaging/test_opaque_tensor_order.py::test_saved_fourth_piece_order_enclosures_and_cost'],'tensor-order':['tests/packaging/test_opaque_tensor_order.py::test_both_orders_enclose_independent_material_integrals','tests/packaging/test_opaque_tensor_order.py::test_physical_axis_callbacks_and_nonunit_polynomial','tests/packaging/test_opaque_tensor_order.py::test_invalid_order_refused_before_field_evaluation','tests/packaging/test_opaque_tensor_order.py::test_original_profile_uses_qualified_fourth_piece_order'],'d345-diagnostic':['tests/packaging/test_opaque_d345_diagnostic.py'],'interval-copy-controls':['tests/packaging/test_opaque_denominator_reuse.py::test_existing_interval_operands_do_not_need_temporary_copies','tests/packaging/test_opaque_denominator_reuse.py::test_zero_addition_keeps_its_fresh_return_contract'],'denominator-cost':['tests/packaging/test_opaque_denominator_cost.py'],'denominator-regression':['tests/packaging/test_opaque_signed_partition.py','tests/packaging/test_opaque_smooth_base.py','tests/packaging/test_opaque_requested_kernel.py','tests/packaging/test_opaque_square_reference.py'],'d343-diagnostic':['tests/packaging/test_opaque_d343_diagnostic.py'],'denominator-reuse':['tests/packaging/test_opaque_denominator_reuse.py','tests/packaging/test_opaque_interval_work.py'],'guidance-cost':['tests/packaging/test_opaque_refinement_guidance.py::test_guidance_cost_on_same_saved_regions'],'refinement-binding':['tests/packaging/test_opaque_refinement_guidance.py::test_original_checker_supplies_separate_refinement_field'],'refinement-guidance':['tests/packaging/test_opaque_refinement_guidance.py'],'signed-partition-diagnostic':['tests/packaging/test_opaque_signed_partition_diagnostic.py'],'signed-partition-additional':['tests/packaging/test_opaque_segmented_remainder.py','tests/packaging/test_opaque_smooth_base.py'],'signed-partition-binding':['tests/packaging/test_opaque_signed_partition.py::test_material_producer_supplies_signed_integrated_bound'],'signed-partition':['tests/packaging/test_opaque_signed_partition.py'],'split-response':['tests/packaging/test_opaque_split_response.py'],'smooth-base-final':['tests/packaging/test_opaque_smooth_base.py','tests/packaging/test_opaque_smooth_branch.py','tests/packaging/test_opaque_segmented_remainder.py::test_full_original_producer_selects_qualified_segmented_checker'],'smooth-base-diagnostic':['tests/packaging/test_opaque_smooth_base_diagnostic.py'],'smooth-base-binding':['tests/packaging/test_opaque_smooth_base.py::test_original_base_and_correction_share_the_qualified_branch_cache'],'smooth-base':['tests/packaging/test_opaque_smooth_base.py'],'arithmetic-oracles':['tests/packaging/test_opaque_shading_intervals.py','tests/packaging/test_opaque_square_reference.py','tests/packaging/test_opaque_moving_reference.py','tests/packaging/test_opaque_curved_reference.py','tests/packaging/test_opaque_requested_kernel.py'],'interval-work':['tests/packaging/test_opaque_interval_work.py'],'d338-diagnostic':['tests/packaging/test_opaque_d338_diagnostic.py'],'requested-kernel':['tests/packaging/test_opaque_requested_kernel.py'],'cold-kernel-cost':['tests/packaging/test_opaque_kernel_cost.py'],'input-reuse':['tests/packaging/test_opaque_input_reuse.py'],'piece1-matched-cost':['tests/packaging/test_opaque_piece1_cost.py'],'segment-binding':['tests/packaging/test_opaque_segmented_remainder.py::test_full_original_producer_selects_qualified_segmented_checker'],'segment-additional':['tests/packaging/test_opaque_segmented_remainder.py::test_segmented_curved_material_retains_jacobian_and_exact_integral','tests/packaging/test_opaque_segmented_remainder.py::test_signed_ordinary_refuses_disjoint_evidence','tests/packaging/test_opaque_segmented_remainder.py::test_full_original_producer_selects_qualified_segmented_checker'],'third-piece-segmented':['tests/packaging/test_opaque_third_piece_diagnostic.py::test_saved_third_piece_segmented_remainder'],'segment-regression':['tests/packaging/test_opaque_peano.py','tests/packaging/test_opaque_localized_peano.py','tests/packaging/test_opaque_signed_peano.py','tests/packaging/test_opaque_transverse_peano.py','tests/packaging/test_opaque_smooth_peano.py','tests/packaging/test_opaque_endpoint_peano.py','tests/packaging/test_opaque_conditioned_center.py','tests/packaging/test_opaque_material_reuse.py','tests/packaging/test_opaque_smooth_branch.py','tests/packaging/test_opaque_boundary_diagnostic.py'],'segmented-remainder':['tests/packaging/test_opaque_segmented_remainder.py'],'third-piece-conditioned':['tests/packaging/test_opaque_third_piece_diagnostic.py::test_saved_third_piece_conditioned_dependence'],'third-piece-diagnostic':['tests/packaging/test_opaque_third_piece_diagnostic.py::test_saved_third_piece_terms_directions_and_cost'],'smooth-branch':['tests/packaging/test_opaque_smooth_branch.py'],'material-reuse':['tests/packaging/test_opaque_material_reuse.py'],'boundary-cost-diagnostic':['tests/packaging/test_opaque_boundary_original_diagnostic.py::test_saved_boundary_uncertainty_and_derivative_cost'],'conditioned-center':['tests/packaging/test_opaque_conditioned_center.py'],'largest-leaf-diagnostic':['tests/packaging/test_opaque_boundary_original_diagnostic.py::test_saved_largest_leaf_terms_and_directions'],'smooth-original-diagnostic':['tests/packaging/test_opaque_boundary_original_diagnostic.py::test_saved_original_smooth_components'],'smooth-peano':['tests/packaging/test_opaque_smooth_peano.py'],'endpoint-peano':['tests/packaging/test_opaque_endpoint_peano.py'],'boundary-original-diagnostic':['tests/packaging/test_opaque_boundary_original_diagnostic.py'],'boundary-diagnostic-controls':['tests/packaging/test_opaque_boundary_diagnostic.py'],'transverse-peano':['tests/packaging/test_opaque_transverse_peano.py'],'signed-peano':['tests/packaging/test_opaque_signed_peano.py'],'localized-peano':['tests/packaging/test_opaque_localized_peano.py'],'campaign-contract':['tests/packaging/test_opaque_reference_campaign.py'],'campaign-piece':['tests/packaging/test_opaque_reference_piece.py'],'partition-work':['tests/packaging/test_opaque_full_material_reference.py::test_partition_minimum_work_diagnostic'],'partition-reference':['tests/packaging/test_opaque_full_material_reference.py::test_partitioned_original_material_reference'],'partition-diagnostic':['tests/packaging/test_opaque_full_material_reference.py::test_original_angular_partition_diagnostic'],'branch-partition':['tests/packaging/test_opaque_branch_partition.py'],'first-material-piece':['tests/python/test_opaque_full_original.py::test_first_original_material_piece_matches_independent_certificate'],'opaque-cuda-first-material-piece':['tests/python/test_opaque_full_original.py::test_first_original_material_piece_matches_independent_certificate'],'peano':['tests/packaging/test_opaque_peano.py'],'original-reservation':['tests/python/test_opaque_full_original.py::test_constant_material_reservation_retains_unknown_cut_fallback'],'full-original-reference':['tests/packaging/test_opaque_full_material_reference.py::test_full_original_material_reference'],'original-residual-diagnostic':['tests/packaging/test_opaque_full_material_reference.py::test_original_residual_error_components'],'square-reference':['tests/packaging/test_opaque_square_reference.py'],'full-original-extract':['tests/python/test_opaque_full_original.py::test_extract_complete_original_material'],'full-original-pixel':['tests/python/test_opaque_full_original.py::test_complete_original_material_pixel'],'opaque-cuda-full-original-pixel':['tests/python/test_opaque_full_original.py::test_complete_original_material_pixel'],'curved-frame':['tests/python/test_opaque_curved_frames.py'],'opaque-cuda-curved-frame':['tests/python/test_opaque_curved_frames.py'],'curved-pixel-reference':['tests/packaging/test_opaque_curved_pixel_reference.py'],'curved-pixels':['tests/python/test_opaque_curved_pixels.py'],'opaque-cuda-curved-pixels':['tests/python/test_opaque_curved_pixels.py'],'native-curved':['tests/python/test_native_opaque_curved.py'],'opaque-cuda-native-curved':['tests/python/test_native_opaque_curved.py'],'curved-original':['tests/packaging/test_opaque_curved_original.py'],'curved-reference':['tests/packaging/test_opaque_curved_reference.py'],'curved-preflight':['tests/python/test_opaque_curved_preflight.py'],'opaque-cuda-camera-regression':['tests/python/test_camera.py'],'camera-regression':['tests/python/test_camera.py'],'frame-execution-controls':['tests/python/test_opaque_frame_execution.py::test_frame_executor_refuses_mutation_cancellation_and_corrupt_storage'],'frame-execution':['tests/python/test_opaque_frame_execution.py::test_complete_frame_execution_with_ordered_storage_and_vjp'],'opaque-cuda-execution':['tests/python/test_opaque_frame_execution.py::test_complete_frame_execution_with_ordered_storage_and_vjp'],'opaque-cuda-knot-exact':['tests/python/test_opaque_knot_exact.py'],'opaque-window':['tests/python/test_opaque_window.py'],'opaque-cuda-projection-exact':['tests/python/test_opaque_exact_diagnostic.py::test_projection_stages_are_device_exact'],'opaque-cuda-exact-diagnostic':['tests/python/test_opaque_exact_diagnostic.py'],'scene-horizon':['tests/python/test_opaque_scene_horizons.py'],'opaque-cuda-scene-horizon':['tests/python/test_opaque_scene_horizons.py::test_production_view_horizon_matches_independent_pixel_and_motion','tests/python/test_opaque_scene_horizons.py::test_production_equal_vertex_normals_keep_quadratic_motion'],'frame-cost':['tests/python/test_opaque_frame_cost.py'],'frame-replay':['tests/python/test_opaque_frame_replay.py'],'opaque-cuda-frame':['tests/python/test_opaque_frame_replay.py'],'frame-storage':['tests/packaging/test_opaque_frame_store.py'],'horizon-material':['tests/python/test_opaque_horizon_transport.py'],'opaque-cuda-horizon':['tests/python/test_opaque_horizon_transport.py'],'knot-geometry':['tests/python/test_opaque_knot_device.py'],'opaque-cuda-knot':['tests/python/test_opaque_knot_device.py'],'knot-material':['tests/python/test_opaque_knot_material.py'],'knot-transport':['tests/packaging/test_opaque_knot_transport.py'],'mesh-evidence':['tests/packaging/test_opaque_mesh_evidence.py'],'mesh-reference-controls':['tests/packaging/test_opaque_mesh_preflight.py','tests/packaging/test_opaque_loss_extension.py','tests/packaging/test_opaque_shading_intervals.py','tests/python/test_opaque_loss_extension_material.py'],'loss-material':['tests/python/test_opaque_loss_extension_material.py'],'loss-polynomial':['tests/packaging/test_opaque_loss_extension.py'],'loss-extension':['tests/packaging/test_opaque_shading_intervals.py::test_interval_quadrature_adds_independent_signed_correction_and_tangent','tests/packaging/test_opaque_shading_intervals.py::test_interval_quadrature_bounds_a_nonsmooth_residual_without_differentiating_it'],'mesh-preflight':['tests/packaging/test_opaque_mesh_preflight.py'],'opaque-cuda-diagnostic':['tests/python/test_opaque_cuda_diagnostic.py'], 'ptx-embedding':['tests/packaging/test_ptx_embedding.py'], 'material-broad':['tests/python/test_opaque_material_branches.py'], 'opaque-cuda':['tests/python/test_opaque_cuda.py'], 'opaque-stream':['tests/python/test_opaque_streaming.py'], 'scene-demo':['tests/python/test_opaque_nonlinear_scene.py::test_opaque_scene_saves_texture_roughness_demonstration'], 'scene-parameters':['tests/python/test_opaque_nonlinear_scene.py::test_opaque_scene_independent_parameter_directions','tests/python/test_opaque_nonlinear_scene.py::test_opaque_scene_saves_texture_roughness_demonstration'], 'nonlinear-scene':['tests/python/test_opaque_nonlinear_scene.py'], 'quadrature':['tests/python/test_native_opaque_quadrature.py'], 'material-texture':['tests/python/test_opaque_material_principled.py::test_textured_material_transition_matches_exact_formula_and_installed_calls'], 'material-branch':['tests/python/test_opaque_material_principled.py::test_roughness_branch_diagnostic'], 'material-oracle':['tests/packaging/test_opaque_material_reference.py'], 'material-principled':['tests/python/test_opaque_material_principled.py::test_roughness_point_and_first_derivative_match_independent_formula','tests/python/test_opaque_material_principled.py::test_spatial_roughness_has_separate_value_and_gradient_error_bounds'], 'moving-principled':['tests/python/test_opaque_moving_principled.py'], 'moving-oracle':['tests/packaging/test_opaque_moving_reference.py'], 'principled-comparison':['tests/python/test_opaque_principled_comparison.py'], 'principled-integral':['tests/python/test_opaque_principled_integral.py'], 'nonlinear-oracle':['tests/packaging/test_opaque_shading_intervals.py'], 'affine-oracle':['tests/packaging/test_opaque_affine_oracle.py','tests/packaging/test_opaque_pixel_oracle.py'], 'colour':['tests/python/test_native_opaque_colour.py'], 'colour-final':['tests/python/test_native_opaque_colour.py','tests/python/test_native_opaque_coverage.py','tests/python/test_opaque_scene.py','tests/python/test_deferred.py','tests/python/test_opaque_visibility.py'], 'translation':['tests/python/test_native_opaque_coverage.py::test_native_opaque_small_area_survives_large_pixel_origin'], 'final':['tests/python/test_native_opaque_coverage.py','tests/python/test_opaque_scene.py','tests/python/test_deferred.py','tests/python/test_opaque_visibility.py'], 'scene':['tests/python/test_opaque_scene.py'], 'regression':['tests/python/test_deferred.py','tests/python/test_opaque_visibility.py'], 'native':['tests/python/test_native_opaque_coverage.py'], 'teaching':['tests/python/test_textbook.py','tests/python/test_m1_lessons.py','tests/python/test_deferred_lessons.py','tests/python/test_gallery_cached.py'], 'comparison':[], 'models':['tests/packaging/test_opaque_boundary_models.py'],
       'models-red':['tests/packaging/test_opaque_boundary_models.py::test_compiled_complete_model_matches_exact_reference[diagonal_a]'],
       'oracle':['tests/packaging/test_opaque_pixel_oracle.py'],
       'oracle-red':['tests/packaging/test_opaque_pixel_oracle.py::test_subdivision_and_interior_vertex_motion_do_not_change_opaque_union'],
       'coverage-red':['tests/python/test_opaque_visibility.py']}


MODES['kernel-regression']=MODES['segment-regression']+MODES['segmented-remainder']+MODES['input-reuse']
MODES['requested-kernel-cost']=['tests/packaging/test_opaque_requested_kernel_cost.py']

MODES['kernel-compat']=['tests/packaging/test_opaque_conditioned_center.py::test_center_and_kernel_reuse_preserve_exact_acceptance_and_charge_work']

MODES['kernel-lazy-controls']=['tests/packaging/test_opaque_requested_kernel.py::test_requested_orders_share_one_position_entry_and_fill_only_missing_orders','tests/packaging/test_opaque_requested_kernel.py::test_failed_order_fill_keeps_other_completed_orders']

MODES['offline-leaf-diagnostic']=['tests/packaging/test_opaque_offline_leaf_diagnostic.py']
MODES['late-boundary-controls']=['tests/packaging/test_opaque_late_boundary.py']
MODES['extended-campaign-piece']=['tests/packaging/test_opaque_extended_piece.py']
MODES['extended-campaign-controls']=['tests/packaging/test_opaque_extended_campaign.py','tests/packaging/test_opaque_reference_campaign.py']
MODES['offline-late-preflight']=['tests/packaging/test_opaque_offline_late_preflight.py']
MODES['offline-preflight']=['tests/packaging/test_opaque_offline_preflight.py']
MODES['offline-regression']=['tests/packaging/test_opaque_square_reference.py','tests/packaging/test_opaque_direct_aggregate.py','tests/packaging/test_opaque_decimal_native.py','tests/packaging/test_opaque_decimal_material.py','tests/packaging/test_opaque_seeded_cover.py']
MODES['offline-profile-controls']=['tests/packaging/test_opaque_offline_profile.py']

MODES['translation-transport-controls']=['tests/packaging/test_opaque_translation_transport.py']

MODES['translation-knot']=['tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[knot-field-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[knot-field-normal]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[knot-geometry-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[knot-geometry-normal]']
MODES['opaque-cuda-translation-knot']=['tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[knot-field-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[knot-field-normal]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[knot-geometry-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[knot-geometry-normal]']
MODES['translation-horizon']=['tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[horizon-field-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[horizon-field-normal]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[horizon-geometry-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[horizon-geometry-normal]']
MODES['opaque-cuda-translation-horizon']=['tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[horizon-field-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[horizon-field-normal]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[horizon-geometry-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[horizon-geometry-normal]']
MODES['translation-visibility']=['tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[visibility-geometry-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[visibility-geometry-normal]']
MODES['opaque-cuda-translation-visibility']=['tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[visibility-geometry-view]', 'tests/python/test_opaque_translation_transport.py::test_original_material_translation_derivative[visibility-geometry-normal]']

MODES['translation-capacity']=['tests/python/test_opaque_composite_capacity.py']
MODES['opaque-cuda-translation-capacity']=['tests/python/test_opaque_composite_capacity.py']
MODES['translation-composite-controls']=['tests/python/test_opaque_composite_probe_controls.py']
MODES['translation-asymmetric-probe']=['tests/python/test_opaque_composite_probe.py::test_asymmetric_scene_radial_diagnostic']
MODES['translation-graded-probe']=['tests/python/test_opaque_composite_probe.py::test_graded_scene_radial_diagnostic']
MODES['translation-composite-probe']=['tests/python/test_opaque_composite_probe.py']
MODES['translation-camera-chain']=['tests/python/test_opaque_original_leaf.py::test_original_camera_chain']
MODES['opaque-cuda-translation-camera-chain']=['tests/python/test_opaque_original_leaf.py::test_original_camera_chain']
MODES['translation-camera-inverse']=['tests/python/test_camera_inverse_order.py']
MODES['opaque-cuda-translation-camera-inverse']=['tests/python/test_camera_inverse_order.py']
MODES['original-motion-installed']=['tests/python/test_opaque_piece_motion_acceptance.py::test_original_motion_meets_independent_piece_endpoints']
MODES['opaque-cuda-original-motion-installed']=MODES['original-motion-installed']
MODES['motion-tangent-controls']=['tests/packaging/test_opaque_motion_tangent.py::test_curved_range_fallback_contains_closed_two_crossing_integrals','tests/packaging/test_opaque_motion_tangent.py::test_first_linear_log_join_keeps_ordinary_tangent_and_measure','tests/packaging/test_opaque_motion_tangent.py::test_curved_map_cannot_borrow_polynomial_crossing_count','tests/packaging/test_opaque_motion_tangent.py::test_first_segment_retains_tiny_tangent_at_zero_cosine','tests/packaging/test_opaque_motion_tangent.py::test_first_join_parameter_flux_encloses_known_discontinuous_loss']
MODES['arb-horizon-domain']=['tests/packaging/test_opaque_motion_tangent.py::test_original_horizon_domain_preflight']
MODES['arb-horizon-chart-diagnostic']=['tests/packaging/test_opaque_motion_tangent.py::test_saved_horizon_transport_chart_bounds']
MODES['motion-map-controls']=['tests/packaging/test_opaque_motion_tangent.py::test_unflattened_original_chart_preserves_angles_tangents_and_measure','tests/packaging/test_opaque_motion_tangent.py::test_unflattened_flux_removes_coordinate_zero_without_discarding_transport','tests/packaging/test_opaque_motion_tangent.py::test_original_motion_fields_match_independent_vector_chain','tests/packaging/test_opaque_motion_tangent.py::test_original_curved_map_keeps_moving_horizon_zero']
MODES['arb-original-motion-reference']=['tests/packaging/test_opaque_motion_tangent.py::test_selected_original_motion_reference']
MODES['original-motion-inventory']=['tests/packaging/test_opaque_motion_inventory.py']
MODES['tangent-original-installed']=['tests/python/test_opaque_original_tangent.py']
MODES['opaque-cuda-tangent-original-installed']=MODES['tangent-original-installed']
MODES['tangent-checker-regression']=['tests/packaging/test_opaque_tangent_flux.py','tests/packaging/test_opaque_tangent_peano.py','tests/packaging/test_opaque_square_reference.py','tests/packaging/test_opaque_knot_transport.py','tests/packaging/test_opaque_peano.py','tests/packaging/test_opaque_signed_peano.py','tests/packaging/test_opaque_localized_peano.py','tests/packaging/test_opaque_seeded_cover.py','tests/packaging/test_opaque_original_tangent.py::test_cyclic_maps_retain_physical_triangle_and_tangent','tests/packaging/test_opaque_original_tangent.py::test_radial_normal_tangent_matches_independent_decimal_vectors']
MODES['knot-partition-controls']=['tests/packaging/test_opaque_knot_partition.py::'+n for n in ['test_knot_cut_targets_both_axis_orientations_with_bounded_work','test_knot_guidance_cannot_invent_root_or_swallow_stop','test_exact_cut_integrates_full_measure_and_moving_jump','test_invalid_cut_is_refused_without_budget_or_acceptance_bypass','test_valid_cut_still_obeys_existing_cell_and_depth_limits']]
MODES['knot-partition-controls'] += ['tests/packaging/test_opaque_knot_partition.py::test_cut_integrates_nonlinear_field_and_complete_moving_measure','tests/packaging/test_opaque_knot_partition.py::test_original_knot_profile_requires_domain_qualification_before_trial']
MODES['arb-boundary-parent']=['tests/packaging/test_opaque_knot_partition.py::test_saved_boundary_parent_targets']
MODES['arb-original-entry-chart']=['tests/packaging/test_opaque_knot_partition.py::test_saved_original_entry_chart']
MODES['angular-cover-controls']=['tests/packaging/test_opaque_knot_partition.py::'+n for n in ['test_monotone_cover_closed_measure_jump_and_transport','test_monotone_cover_two_exits_and_closed_nonlinear_measure','test_monotone_cover_refuses_invalid_topology_budget_and_chart']]
MODES['angular-cover-controls'] += ['tests/packaging/test_opaque_knot_partition.py::test_angular_cover_refuses_unqualified_accuracy','tests/packaging/test_opaque_knot_partition.py::test_monotone_cover_retains_thin_positive_band']
MODES['arb-sqrt-positive-controls']=['tests/packaging/test_opaque_knot_partition.py::test_arb_sqrt_keeps_proved_positive_constant_and_mixed_coefficients']
MODES['arb-tight-discriminant-controls']=['tests/packaging/test_opaque_knot_partition.py::test_tight_discriminant_preserves_closed_curved_area_and_motion','tests/packaging/test_opaque_knot_partition.py::test_tight_original_discriminant_point_and_motion_control']
MODES['arb-tight-initial-bounds']=['tests/packaging/test_opaque_knot_partition.py::test_tight_original_initial_bound_diagnostic']
MODES['arb-tight-chart-0']=['tests/packaging/test_opaque_knot_partition.py::test_tight_original_chart_integral[0]']
MODES['arb-tight-chart-27']=['tests/packaging/test_opaque_knot_partition.py::test_tight_original_chart_integral[27]']
MODES['arb-tight-chart-32']=['tests/packaging/test_opaque_knot_partition.py::test_tight_original_chart_integral[32]']
MODES['arb-tight-regression']=['tests/packaging/test_opaque_curved_reference.py','tests/packaging/test_opaque_knot_partition.py::test_original_material_chart_state_and_frozen_stage_equivalence','tests/packaging/test_opaque_knot_partition.py::test_original_material_shared_driver_keeps_construction_deadline']
MODES['arb-tight-shared-controls']=['tests/packaging/test_opaque_knot_partition.py::test_tight_shared_cover_matches_frozen_local_stages','tests/packaging/test_opaque_knot_partition.py::test_tight_shared_driver_keeps_explicit_selection_and_construction_deadline']
MODES['arb-tight-shared-piece']=['tests/packaging/test_opaque_knot_partition.py::test_tight_shared_original_complete_piece']
MODES['arb-horizon-cache-controls']=['tests/packaging/test_opaque_knot_partition.py::test_horizon_cache_exact_keys_ownership_backend_and_bound']
MODES['arb-tight-cell-cost']=['tests/packaging/test_opaque_knot_partition.py::test_tight_shared_initial_cell_cost_diagnostic']
MODES['arb-prepared-angle-controls']=['tests/packaging/test_opaque_knot_partition.py::test_prepared_angle_preserves_original_high_mixed_and_partial','tests/packaging/test_opaque_knot_partition.py::test_prepared_angle_rejects_mismatched_jets_and_crosses_backend_safely']
MODES['arb-material-cache-controls']=['tests/packaging/test_opaque_knot_partition.py::test_material_input_cache_keys_ownership_context_and_bound','tests/packaging/test_opaque_knot_partition.py::test_material_input_cache_preserves_frozen_chart_after_prepare_and_interleave']
MODES['arb-algebraic-angle-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::'+n for n in ['test_residual_inverse_retains_closed_high_and_motion_coefficients','test_residual_inverse_refuses_uncertified_slope_and_bad_contract','test_original_quartic_residual_encloses_direct_vector_identity']]
MODES['arb-algebraic-cover-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_residual_cover_keeps_complete_moving_measure_and_tiny_terms','tests/packaging/test_opaque_algebraic_angle_inverse.py::test_algebraic_selector_is_explicit_and_keeps_construction_deadline']
MODES['arb-algebraic-slope-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_quartic_root_slope_identity_matches_direct_positive_derivative']
MODES['arb-algebraic-exit-diagnostic']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_saved_exit_transport_diagnostic_compares_both_inverses']
MODES['mapped-flux-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_mapped_flux_keeps_collapsing_chart_jump_and_tiny_measure','tests/packaging/test_opaque_algebraic_angle_inverse.py::test_mapped_flux_refuses_missing_physical_monotonicity_or_measure']
MODES['mapped-flux-weight-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_material_flux_override_preserves_both_roles_and_weights_measure_once']
MODES['arb-mapped-exit-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_saved_exit_mapped_transport_retains_finite_complete_correction']
MODES['arb-mapped-exit-local']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_algebraic_mapped_original_chart66_integrates']
MODES['arb-positive-exit-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_positive_exit_angle_keeps_closed_inverse_and_complete_mixed_measure','tests/packaging/test_opaque_algebraic_angle_inverse.py::test_positive_exit_angle_cover_integrates_all_moving_measure']
MODES['arb-positive-exit-local']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_positive_exit_angle_original_chart66_integrates']
MODES['arb-positive-exit-diagnostic']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_positive_exit_largest_leaf_error_diagnostic']
MODES['arb-normal-light-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_normal_light_spatial_bound_matches_independent_normalization']
MODES['arb-normal-tangent-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_normal_light_tangent_bound_keeps_intrinsic_and_coordinate_motion']
MODES['arb-exit-hint-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_exit_chart_forwards_current_residual_split_hint','tests/packaging/test_opaque_square_reference.py']
MODES['arb-algebraic-angle-local']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_algebraic_shared_original_chart27_integrates']
MODES['arb-algebraic-angle-piece']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_algebraic_shared_original_complete_piece']
MODES['arb-value-bracket-controls']=['tests/packaging/test_opaque_knot_partition.py::test_prepared_value_only_brackets_match_exact_spatial_endpoints','tests/packaging/test_opaque_knot_partition.py::test_inverse_value_only_path_keeps_closed_mixed_derivatives']
MODES['arb-quadratic-sqrt-controls']=['tests/packaging/test_opaque_knot_partition.py::'+n for n in ['test_quadratic_sqrt_keeps_exact_square_and_moving_composition','test_quadratic_sqrt_nonlinear_mixed_matches_point_algebra','test_quadratic_sqrt_refuses_nonpositive_domain_and_keeps_tiny_curvature']]
MODES['arb-shared-initial-bounds']=['tests/packaging/test_opaque_knot_partition.py::test_shared_original_initial_bound_diagnostic']
MODES['arb-original-shared-controls']=['tests/packaging/test_opaque_knot_partition.py::test_original_material_chart_state_and_frozen_stage_equivalence','tests/packaging/test_opaque_knot_partition.py::test_original_material_shared_driver_keeps_construction_deadline']
MODES['arb-original-shared-piece']=['tests/packaging/test_opaque_knot_partition.py::test_original_material_shared_driver_complete_piece']
MODES['arb-hybrid-stage-preflight']=['tests/packaging/test_opaque_knot_partition.py::test_original_hybrid_whole_chart_stage_preflight']
MODES['shared-chart-controls']=['tests/packaging/test_opaque_knot_partition.py::'+n for n in ['test_shared_charts_sum_complete_measure_and_transport','test_shared_charts_refuse_sum_of_local_passes','test_shared_charts_never_reset_work_depth_or_deadline','test_shared_charts_validate_cover_before_work']]
MODES['shared-chart-controls'] += ['tests/packaging/test_opaque_knot_partition.py::test_shared_charts_keep_hybrid_edges_jumps_and_tiny_measure','tests/packaging/test_opaque_knot_partition.py::test_shared_charts_refuse_missing_channel']
MODES['arb-shared-chart-controls']=MODES['shared-chart-controls']
MODES['arb-angle-only-controls']=['tests/packaging/test_opaque_knot_partition.py::test_angle_only_matches_all_saved_coefficients','tests/packaging/test_opaque_knot_partition.py::test_angle_only_refuses_unsupported_profile']
MODES['arb-angle-coordinate-factors']=['tests/packaging/test_opaque_knot_partition.py::test_angle_coordinate_original_factor_bounds']
MODES['arb-angle-coordinate-integral']=['tests/packaging/test_opaque_knot_partition.py::test_angle_coordinate_original_chart_integration']
MODES['arb-angle-cover-original']=['tests/packaging/test_opaque_knot_partition.py::test_complete_original_hybrid_angle_cover']
MODES['arb-angle-cover-controls']=['tests/packaging/test_opaque_knot_partition.py::test_angle_cover_complete_measure_jump_and_transport','tests/packaging/test_opaque_knot_partition.py::test_angle_cover_two_exits_nonlinear_measure']
MODES['angle-coordinate-controls']=['tests/packaging/test_opaque_knot_partition.py::'+n for n in ['test_angle_coordinate_closed_measure_and_transport','test_angle_coordinate_nonlinear_inverse_high_mixed','test_angle_coordinate_refuses_exit_and_nonpositive_slope','test_original_radial_angle_derivative_matches_independent_seed']]
MODES['arb-angle-coordinate-controls']=MODES['angle-coordinate-controls']
MODES['arb-centered-factor-bounds']=['tests/packaging/test_opaque_knot_partition.py::test_centered_chart_factor_bounds']
MODES['arb-centered-loss-controls']=['tests/packaging/test_opaque_loss_extension.py']
MODES['arb-angular-high-bounds']=['tests/packaging/test_opaque_knot_partition.py::test_saved_angular_chart_high_bounds_are_finite']
MODES['arb-angular-chart-diagnostic']=['tests/packaging/test_opaque_knot_partition.py::test_saved_angular_chart_stage_diagnostic']
MODES['arb-angular-chart-integration']=['tests/packaging/test_opaque_knot_partition.py::test_original_angular_chart_integration']
MODES['arb-complete-angular-cover']=['tests/packaging/test_opaque_knot_partition.py::test_complete_original_angular_cover']
MODES['arb-angular-cover-controls']=MODES['angular-cover-controls']
MODES['arb-angular-cost-audit']=['tests/packaging/test_opaque_knot_partition.py::test_original_angular_cover_cost_audit']
MODES['arb-entry-knot-cover']=['tests/packaging/test_opaque_knot_partition.py::test_original_entry_following_domain_cover']
MODES['knot-entry-controls']=['tests/packaging/test_opaque_knot_partition.py::test_entry_chart_full_cover_and_moving_flux_have_closed_integrals','tests/packaging/test_opaque_knot_partition.py::test_entry_chart_refuses_unsupported_topology_and_outside_chart']
MODES['knot-entry-controls'].append('tests/packaging/test_opaque_knot_partition.py::test_entry_chart_nonlinear_inverse_root_integral')
MODES['knot-entry-controls'].append('tests/packaging/test_opaque_knot_partition.py::test_entry_cover_cannot_enable_unqualified_accuracy')
MODES['arb-knot-entry-controls']=MODES['knot-entry-controls']
MODES['arb-knot-edge-events']=['tests/packaging/test_opaque_knot_partition.py::test_saved_original_knot_edge_events']
MODES['knot-edge-controls']=['tests/packaging/test_opaque_knot_partition.py::test_knot_edge_crossing_locations_and_parameter_motion']
MODES['knot-endpoint-controls']=['tests/packaging/test_opaque_knot_partition.py::test_uniform_knot_endpoints_use_proved_transverse_monotonicity','tests/packaging/test_opaque_knot_partition.py::test_uniform_knot_endpoint_refinement_cannot_hide_interior_crossing']
MODES['arb-curved-knot-cover']=['tests/packaging/test_opaque_knot_partition.py::test_original_curved_knot_domain_cover']
MODES['arb-original-band']=['tests/packaging/test_opaque_knot_partition.py::test_original_interior_angular_band']
MODES['knot-band-controls']=['tests/packaging/test_opaque_knot_partition.py::test_knot_band_proves_width_when_root_ranges_overlap','tests/packaging/test_opaque_knot_partition.py::test_knot_band_nonlinear_integral_preserves_parameter_conservation']
MODES['knot-band-controls'].append('tests/packaging/test_opaque_knot_partition.py::test_curved_knot_cover_cannot_enable_unqualified_accuracy_trial')
MODES['arb-knot-band-controls']=MODES['knot-band-controls']
MODES['knot-region-controls']=['tests/packaging/test_opaque_knot_partition.py::'+n for n in ['test_knot_region_high_mixed_derivatives_match_closed_sqrt','test_knot_region_uniform_bracket_and_boundary_refusals','test_knot_region_full_integrals_measure_and_shared_boundary_flux','test_knot_region_varying_physical_jacobian_and_jump_integral']]
MODES['knot-root-controls']=['tests/packaging/test_opaque_knot_partition.py::test_curved_knot_root_and_motion_match_closed_implicit_derivatives','tests/packaging/test_opaque_knot_partition.py::test_curved_knot_root_refuses_unproved_monotonicity_and_keeps_tiny_root']
MODES['arb-knot-region-diagnostic']=['tests/packaging/test_opaque_knot_partition.py::test_saved_original_whole_knot_region']
MODES['arb-knot-region-controls']=MODES['knot-region-controls']
MODES['arb-knot-contour-diagnostic']=['tests/packaging/test_opaque_knot_partition.py::test_saved_original_knot_contour_slices']
MODES['arb-knot-cut-diagnostic']=['tests/packaging/test_opaque_knot_partition.py::test_saved_original_knot_cut_diagnostics']
MODES['motion-subdivision-controls']=['tests/packaging/test_opaque_motion_subdivision.py::test_one_monotone_axis_uses_edges_not_unproved_corners','tests/packaging/test_opaque_motion_subdivision.py::test_unproved_axes_preserve_interior_extremum_and_parameter_tangent']
MODES['motion-subdivision-controls'].append('tests/packaging/test_opaque_motion_subdivision.py::test_horizon_split_retains_exact_partition_and_ordinary_midpoints')
MODES['motion-subdivision-controls'].append('tests/packaging/test_opaque_motion_subdivision.py::test_unqualified_graded_integrator_is_refused_before_input_access')
MODES['motion-subdivision-controls'].append('tests/packaging/test_opaque_motion_subdivision.py::test_partial_monotonicity_requires_explicit_flag')
MODES['arb-subdivision-diagnostic']=['tests/packaging/test_opaque_motion_subdivision.py::test_saved_segment_subdivision_diagnostic']
MODES['motion-final-controls']=MODES['tangent-checker-regression']+MODES['motion-tangent-controls']+MODES['motion-map-controls']+['tests/packaging/test_opaque_curved_reference.py','tests/packaging/test_opaque_moving_reference.py']
MODES['subdivision-final-controls']=MODES['motion-final-controls']+MODES['motion-subdivision-controls']+['tests/packaging/test_opaque_monotone_bounds.py','tests/packaging/test_opaque_loss_extension.py']
MODES['knot-band-final']=MODES['subdivision-final-controls']+MODES['knot-partition-controls']+MODES['knot-root-controls']+MODES['knot-region-controls']+MODES['knot-band-controls']+['tests/packaging/test_opaque_square_reference.py']
MODES['knot-boundary-final']=MODES['knot-band-final']+MODES['knot-endpoint-controls']+MODES['knot-edge-controls']
MODES['angular-cover-final']=MODES['knot-boundary-final']+MODES['knot-entry-controls']+MODES['angular-cover-controls']
MODES['dashboard-cache-controls']=['tests/python/test_testing_dashboard.py::test_catalog_cache_invalidates_new_evidence_and_keeps_live_staleness','tests/python/test_testing_dashboard.py::test_page_selects_latest_relevant_execution_not_latest_success','tests/python/test_testing_dashboard.py::test_page_pagination_counts_are_local_and_filters_do_not_change_summary','tests/python/test_testing_dashboard.py::test_demo_page_is_permanent_and_independent_of_test_execution','tests/python/test_testing_dashboard.py::test_page_api_requires_no_run_and_keeps_general_dashboard']
MODES['arb-centered-final']=MODES['angular-cover-final']
MODES['knot-entry-final']=MODES['knot-boundary-final']+MODES['knot-entry-controls']
MODES['arb-knot-boundary-controls']=MODES['knot-endpoint-controls']+MODES['knot-edge-controls']
MODES['knot-region-final']=MODES['subdivision-final-controls']+MODES['knot-partition-controls']+MODES['knot-root-controls']+MODES['knot-region-controls']+['tests/packaging/test_opaque_square_reference.py']
MODES['knot-partition-final']=MODES['subdivision-final-controls']+MODES['knot-partition-controls']+['tests/packaging/test_opaque_square_reference.py']
MODES['arb-original-tangent-cyclic']=['tests/packaging/test_opaque_original_tangent.py::test_cyclic_original_normal_tangent_certificate']
MODES['arb-tangent-map']=['tests/packaging/test_opaque_original_tangent.py::test_bounded_cyclic_map_comparison']
MODES['tangent-original-controls']=['tests/packaging/test_opaque_original_tangent.py::test_cyclic_maps_retain_physical_triangle_and_tangent','tests/packaging/test_opaque_original_tangent.py::test_radial_normal_tangent_matches_independent_decimal_vectors','tests/packaging/test_opaque_square_reference.py','tests/packaging/test_opaque_knot_transport.py']
MODES['arb-original-tangent']=['tests/packaging/test_opaque_original_tangent.py::test_first_original_normal_tangent_certificate']
MODES['tangent-peano-controls']=['tests/packaging/test_opaque_tangent_peano.py']
MODES['arb-tangent-regions']=['tests/packaging/test_opaque_original_tangent.py::test_saved_residual_regions_use_tighter_validated_tangent_bounds']
MODES['tangent-flux-controls']=['tests/packaging/test_opaque_tangent_flux.py']
MODES['translation-original-leaf']=['tests/python/test_opaque_original_leaf.py::test_original_leaf_adjoints']
MODES['opaque-cuda-translation-original-leaf']=['tests/python/test_opaque_original_leaf.py::test_original_leaf_adjoints']
MODES['translation-boundary-header']=['tests/python/test_opaque_boundary_transport.py::test_native_boundary_diagnostic_header_precedes_edge_validation']
MODES['translation-principal-controls']=['tests/python/test_opaque_principal_transport.py']
MODES['opaque-cuda-translation-principal-controls']=['tests/python/test_opaque_principal_transport.py']
MODES['translation-repair-regression']=['tests/python/test_opaque_scene_horizons.py', 'tests/python/test_native_opaque_quadrature.py', 'tests/python/test_opaque_streaming.py::test_opaque_stream_replayed_gradient_is_tile_partition_independent', 'tests/python/test_opaque_streaming.py::test_opaque_stream_validates_tile_and_total_output_budget', 'tests/python/test_opaque_streaming.py::test_opaque_stream_batched_depth_material_and_subpixel_windows_match_resident', 'tests/python/test_opaque_streaming.py::test_opaque_perspective_normalization_explicit_vjp_gradcheck', 'tests/python/test_native_opaque_curved.py::test_curved_area_boundary_flux_and_complement', 'tests/python/test_native_opaque_curved.py::test_curved_all_geometry_and_coefficient_gradients', 'tests/python/test_native_opaque_curved.py::test_curved_refuses_tangent_uncertified_and_combined_capacity', 'tests/python/test_native_opaque_curved.py::test_curved_radial_refinement_keeps_shared_capacity']
MODES['opaque-cuda-translation-old-curved']=['tests/python/test_native_opaque_curved.py::test_curved_shared_kernel_bytes_and_vjps']
MODES['translation-boundary']=['tests/python/test_opaque_boundary_transport.py']
MODES['opaque-cuda-translation-boundary']=['tests/python/test_opaque_boundary_transport.py']
MODES['translation-loss-transport']=['tests/python/test_opaque_loss_transport.py']
MODES['translation-scene']=['tests/python/test_opaque_translation_scene.py']
MODES['opaque-cuda-translation-scene']=['tests/python/test_opaque_translation_scene.py']

MODES['curved-transport-map']=['tests/packaging/test_opaque_curved_transport_map.py']

MODES['streamed-motion-controls']=['tests/packaging/test_opaque_streamed_motion.py']

MODES.update({'translation-motion-refusal':['tests/python/test_opaque_motion_refusal.py'], 'translation-original-motion':['tests/python/test_opaque_original_motion.py::test_original_motion_diagnostic'], 'opaque-cuda-translation-original-motion':['tests/python/test_opaque_original_motion.py::test_original_motion_diagnostic']})

MODES.update({'translation-original-normal-vjp':['tests/python/test_opaque_original_motion.py::test_original_normal_input_vjp'],'opaque-cuda-translation-original-normal-vjp':['tests/python/test_opaque_original_motion.py::test_original_normal_input_vjp']})

MODES.update({'translation-separation-candidate':['tests/python/test_opaque_separation.py']})

MODES.update({'translation-separation-proof':['tests/python/test_opaque_separation.py::test_separation_proof_never_discards_positive_overlap']})

MODES.update({'translation-separation-installed':['tests/python/test_opaque_separation.py::test_installed_separation_retains_tiny_positive_overlap_and_gradient','tests/python/test_opaque_separation.py::test_adapter_preserves_all_registered_dispatch_entries'],'opaque-cuda-translation-separation-installed':['tests/python/test_opaque_separation.py::test_installed_separation_retains_tiny_positive_overlap_and_gradient']})

MODES['arb-145-controls']=sum((MODES[n] for n in ['arb-algebraic-angle-controls','arb-algebraic-cover-controls','arb-algebraic-slope-controls','mapped-flux-controls','mapped-flux-weight-controls','arb-positive-exit-controls','arb-normal-light-controls','arb-normal-tangent-controls','arb-tight-shared-controls','arb-tight-regression','arb-angle-coordinate-controls','arb-angle-cover-controls','knot-transport']),[])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--reference-profile',default='standard')
    args=parser.parse_args()
    from tools.opaque_reference_profile import worker_profile
    limits=worker_profile(args.mode,args.reference_profile)
    cost.PROFILE.update(cooperative_seconds=limits.worker_seconds,watchdog_seconds=limits.watchdog_seconds,sampled_working_set_limit_bytes=limits.host_bytes)
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if args.mode.startswith('opaque-cuda'):cost.PROFILE['sampled_working_set_limit_bytes']=2*1024**3
    if not args.worker:
        return cost.supervise(args.fixtures,args.output,worker_module='tools.opaque_coverage_probe',worker_args=('--mode',args.mode,'--reference-profile',args.reference_profile))
    started=time.perf_counter()
    report=dict(status='working',worker_pid=os.getpid(),mode=args.mode,profile=dict(threads=1,host_bytes=cost.PROFILE['sampled_working_set_limit_bytes'],cooperative_seconds=limits.worker_seconds,watchdog_seconds=limits.watchdog_seconds),tests={},reference_profile=limits.record())
    try:
        report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('tools/opaque_smooth_peano.py'),Path('tools/opaque_boundary_diagnostic.py'),Path('tools/opaque_transverse_peano.py'),Path('tools/opaque_reference_campaign.py'),Path('tools/opaque_partitioned_material_reference.py'),Path('tools/opaque_branch_partition.py'),Path('tools/opaque_peano_material.py'),Path('tools/opaque_peano.py'),Path('tools/opaque_full_material_reference.py'),Path('tools/opaque_square_reference.py'),Path('tools/opaque_curved_reference.py'),Path('tools/opaque_material_branches.py'),Path('tools/opaque_material_reference.py'),Path('tools/opaque_moving_reference.py'),Path('tools/opaque_shading_intervals.py'),Path('tools/opaque_principled_reference.py'),Path('tools/opaque_coverage_probe.py'),Path('tools/opaque_pixel_oracle.py'),Path('tools/opaque_boundary_models.py'),Path('tools/opaque_boundary_models.c')] if p.exists()}
        dll=os.environ.get('GFFX_OPAQUE_MODEL_DLL')
        if dll:report['candidate_dll']={'path':dll,'sha256':hashlib.sha256(Path(dll).read_bytes()).hexdigest()}
        import pytest
        if args.mode.endswith('-installed') or 'translation-' in args.mode and args.mode!='translation-transport-controls' or args.mode in ('original-acceptance','opaque-cuda-original-acceptance','original-radial-horizon','original-radial-maps','original-radial','original-subpixel','original-piece-error','first-material-piece','opaque-cuda-first-material-piece','original-reservation','full-original-extract','full-original-pixel','opaque-cuda-full-original-pixel','curved-frame','opaque-cuda-curved-frame','curved-pixels','opaque-cuda-curved-pixels','native-curved','opaque-cuda-native-curved','curved-preflight','opaque-cuda-camera-regression','camera-regression','frame-execution-controls','frame-execution','opaque-cuda-execution','opaque-cuda-knot-exact','opaque-window','opaque-cuda-projection-exact','opaque-cuda-exact-diagnostic','scene-horizon','opaque-cuda-scene-horizon','frame-cost','frame-replay','opaque-cuda-frame','horizon-material','opaque-cuda-horizon','knot-geometry','opaque-cuda-knot','knot-material','mesh-reference-controls','loss-material','opaque-cuda-diagnostic','material-broad','opaque-cuda','opaque-stream','scene-demo','scene-parameters','nonlinear-scene','quadrature','material-texture','material-branch','material-principled','moving-principled','principled-comparison','principled-integral','coverage-red','native','regression','scene','final','translation','colour','colour-final','original-motion-installed','opaque-cuda-original-motion-installed','original-face-motion-installed','opaque-cuda-original-face-motion-installed','flat-face-tensor-diagnostic','flat-refinement-controls','flat-native-controls','opaque-cuda-flat-native-controls','ordered-motion-controls','original-pixel-motion-installed','opaque-cuda-original-pixel-motion-installed'):
            import torch
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            from tools.shading_cuda_closeout import identity
            report['wheel']=identity(Path(os.environ.get('GFFX_OPAQUE_WHEEL',str(WHEEL))),os.environ.get('GFFX_OPAQUE_WHEEL_SHA',SHA))
        for name in MODES[args.mode]:
            p=Path(name.split('::')[0]);report['tests'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if args.mode=='comparison':
            import importlib.util
            path=Path('tests/packaging/test_opaque_boundary_models.py')
            spec=importlib.util.spec_from_file_location('opaque_cases',path)
            cases=importlib.util.module_from_spec(spec);spec.loader.exec_module(cases)
            from tools.opaque_boundary_models import evaluate
            report['fixture_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            report['qualification']='finite C11 research candidates; no upstream renderer execution, production wheel/API, CUDA or reverse-VJP qualification'
            report['cases']=[]
            for name,v,f,rgb,velocity,h,w in cases.CASES:
                cost.check_deadline(started+limits.worker_seconds)
                image,jvp=cases.exact(v,f,rgb,velocity,h,w)
                row=dict(name=name,height=h,width=w,reference_image=[[str(q) for q in p] for p in image],reference_jvp=[[str(q) for q in p] for p in jvp],models={})
                for model in ('complete','sampled_boundary','micro_edge'):
                    result=evaluate(model,v,f,rgb,velocity,height=h,width=w)
                    result['max_image_error']=max(abs(float(a)-b) for x,y in zip(image,result['image']) for a,b in zip(x,y))
                    result['max_jvp_error']=max(abs(float(a)-b) for x,y in zip(jvp,result['derivative']) for a,b in zip(x,y))
                    if model=='complete':
                        assert result['max_image_error']<=1e-12 and result['max_jvp_error']<=1e-12
                    row['models'][model]=result
                report['cases'].append(row)
            report.update(status='passed',seconds=time.perf_counter()-started)
            cost.write_report(args.output,report)
            return 0
        class Deadline:
            def pytest_runtest_setup(self,item):cost.check_deadline(started+limits.worker_seconds)
            def pytest_runtest_teardown(self,item,nextitem):cost.check_deadline(started+limits.worker_seconds)
        selection=['-k','not cuda and not mixed_devices'] if args.mode=='camera-regression' else ['-k','not shared_kernel'] if args.mode=='native-curved' else ['-k','cuda or mixed_devices'] if args.mode=='opaque-cuda-camera-regression' else []
        code=pytest.main(['-q','--tb=short','-p','no:cacheprovider',*MODES[args.mode],*selection,f'--junitxml={args.output.with_suffix(".xml")}'],plugins=[Deadline()])
        if args.mode in ('campaign-piece','extended-campaign-piece'):
            result=Path(os.environ['GFFX_CAMPAIGN_RESULT'])
            report['campaign']=dict(spec_sha256=os.environ['GFFX_CAMPAIGN_SPEC_SHA'],piece=int(os.environ['GFFX_CAMPAIGN_PIECE']),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest() if result.exists() else None)
        report.update(status='passed' if code==0 else 'failed',pytest_exit_code=int(code),seconds=time.perf_counter()-started)
        cost.write_report(args.output,report)
        return 0 if code==0 else 1
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        cost.write_report(args.output,report)
        raise


MODES['arb-exit-error-components']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_exit_correction_error_components']

MODES['arb-outer-edge-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_prepared_outer_edge_matches_direct_normalized_fields']

MODES['arb-repaired-exit-piece']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_repaired_exit_shared_original_complete_piece']

MODES['arb-repaired-remaining-diagnostic']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_repaired_piece_remaining_chart_diagnostic']
MODES['arb-repaired-cost-diagnostic']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_repaired_initial_cell_cost_diagnostic']

MODES['arb-exit-light-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_repaired_exit_light_flux_is_finite','tests/packaging/test_opaque_algebraic_angle_inverse.py::test_mapped_flux_keeps_collapsing_chart_jump_and_tiny_measure','tests/packaging/test_opaque_algebraic_angle_inverse.py::test_material_flux_override_preserves_both_roles_and_weights_measure_once']

MODES['arb-incremental-inverse-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_incremental_quartic_inverse_matches_closed_mixed_coefficients']

MODES['arb-incremental-connection-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_original_incremental_inverse_matches_preserved_recurrence']

MODES['arb-positive-exit68-local']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_positive_exit_angle_original_chart68_integrates']

MODES['arb-exit68-monotonicity']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_exit68_largest_leaf_monotonicity_diagnostic']

MODES['arb-angle-light-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_angle_light_differential_retains_coordinate_and_intrinsic_motion']

MODES['arb-angle-light-connection']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_exit68_light_differential_proves_saved_monotonicity']

MODES['arb-146-controls']=list(dict.fromkeys(MODES['arb-145-controls']+MODES['arb-outer-edge-controls']+MODES['arb-incremental-inverse-controls']+MODES['arb-incremental-connection-controls']+MODES['arb-angle-light-controls']))

MODES['arb-low-angle-diagnostic']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_low_angle_inverse_bound_diagnostic']

MODES['arb-unsquared-inverse-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_inverse_matches_closed_mixed_coefficients']

MODES['arb-unsquared-low-angle']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_low_angle_bound_diagnostic','tests/packaging/test_opaque_algebraic_angle_inverse.py::test_original_unsquared_inverse_matches_preserved_recurrence']

MODES['arb-unsquared-chart27']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_original_chart_integrates[27]']
MODES['arb-unsquared-chart66']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_original_chart_integrates[66]']
MODES['arb-unsquared-chart68']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_original_chart_integrates[68]']
MODES['arb-unsquared-piece']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_shared_original_complete_piece']

MODES['arb-unsquared-regression']=list(dict.fromkeys(MODES['arb-146-controls']+MODES['arb-unsquared-inverse-controls']+['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_original_unsquared_inverse_matches_preserved_recurrence', 'tests/packaging/test_opaque_algebraic_angle_inverse.py::test_original_unsquared_residual_encloses_direct_vector_identity', 'tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_root_slope_identity_matches_direct_positive_derivative', 'tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_selector_requires_qualified_base_and_charges_construction']))

MODES['arb-physical-angle-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_angle_inverse_keeps_closed_moving_radius_and_jacobian']

MODES['arb-physical-cover-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_angle_cover_retains_complete_moving_measure']

MODES['arb-physical-connection']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_original_map_matches_complete_material_jet','tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_selector_and_construction_deadline']

MODES['arb-physical-chart27']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_original_chart_integrates[27]']
MODES['arb-physical-chart57']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_original_chart_integrates[57]']
MODES['arb-physical-chart66']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_original_chart_integrates[66]']
MODES['arb-physical-chart68']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_original_chart_integrates[68]']
MODES['arb-physical-piece']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_shared_original_complete_piece']
MODES['arb-physical-regression']=MODES['arb-unsquared-regression']+MODES['arb-physical-angle-controls']+MODES['arb-physical-cover-controls']+MODES['arb-physical-connection']

MODES['arb-physical-chart63']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_original_chart_integrates[63]']
MODES['arb-physical-chart65']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_original_chart_integrates[65]']

MODES['arb-entry-bound-diagnostic']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_entry_remaining_bound_diagnostic']

MODES['arb-entry-light-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_entry_light_bound_removes_range_fallback']

MODES['arb-physical-polynomial-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_prepared_polynomial_keeps_outward_endpoints_and_context']

MODES['arb-146-final-controls']=MODES['arb-physical-regression']+MODES['arb-entry-light-controls']+MODES['arb-physical-polynomial-controls']

MODES['arb-light-cost-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_material_uses_direct_light_proof_without_repeated_cross_products']

MODES['arb-146-qualified-controls']=MODES['arb-146-final-controls']+MODES['arb-light-cost-controls']

MODES['arb-physical-cost-diagnostic']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_remaining_cell_cost_diagnostic']

MODES['arb-smooth-bound-diagnostic']=['tests/packaging/test_opaque_local_smooth.py::test_smooth_original_bound_diagnostic']

MODES['arb-local-smooth-controls']=['tests/packaging/test_opaque_local_smooth.py::'+name for name in ('test_local_smooth_exact_polynomial_and_motion','test_local_smooth_kernel_masses_match_exact_total_and_symmetry','test_local_smooth_sharp_rational_has_independent_integral','test_local_smooth_explicit_contract_and_deadline')]

MODES['arb-polynomial-row-controls']=['tests/packaging/test_opaque_local_smooth.py::test_prepared_polynomial_reuses_transverse_rows_without_endpoint_change']

MODES['arb-local-smooth-original46']=['tests/packaging/test_opaque_local_smooth.py::test_local_smooth_original_comparison[46]']
MODES['arb-local-smooth-original63']=['tests/packaging/test_opaque_local_smooth.py::test_local_smooth_original_comparison[63]']
MODES['arb-local-smooth-compiled-controls']=['tests/packaging/test_opaque_local_smooth.py::test_local_smooth_arb_exact_controls']

MODES['arb-complete-contributor-controls']=['tests/packaging/test_opaque_local_smooth.py::test_complete_cover_counts_all_contributors_without_local_acceptance']
MODES['arb-transverse-cost']=['tests/packaging/test_opaque_local_smooth.py::test_transverse_reuse_controlled_cost']

MODES['arb-localized-connection']=['tests/packaging/test_opaque_local_smooth.py::test_localized_chart_connection_and_no_implicit_budget_reset']
MODES['arb-localized-whole-piece']=['tests/packaging/test_opaque_local_smooth.py::test_localized_fresh_complete_original_piece']
MODES['arb-147-controls']=MODES['arb-146-qualified-controls']+MODES['arb-local-smooth-controls']+MODES['arb-polynomial-row-controls']+MODES['arb-complete-contributor-controls']+MODES['arb-localized-connection']+['tests/packaging/test_opaque_square_reference.py']

MODES['arb-147-driver-controls']=['tests/packaging/test_opaque_algebraic_angle_inverse.py::'+n for n in ('test_algebraic_selector_is_explicit_and_keeps_construction_deadline','test_unsquared_selector_requires_qualified_base_and_charges_construction','test_physical_selector_and_construction_deadline')]+['tests/packaging/test_opaque_knot_partition.py::'+n for n in ('test_tight_shared_driver_keeps_explicit_selection_and_construction_deadline','test_original_material_shared_driver_keeps_construction_deadline')]+MODES['arb-localized-connection']

MODES['arb-directional-smooth-diagnostic']=['tests/packaging/test_opaque_tensor_smooth.py::test_directional_tensor_remainder_diagnostic']

MODES['arb-tensor-smooth-controls']=['tests/packaging/test_opaque_tensor_smooth.py::'+n for n in ('test_tensor_smooth_exact_integral_motion_and_tiny_terms','test_tensor_smooth_uses_24_samples_and_all_corrections','test_tensor_smooth_rational_integral_and_moving_denominator','test_tensor_smooth_contract_deadline_and_mixed_chart_cover','test_tensor_gauss8_independent_moments_through_fifteen')]

MODES['arb-tensor-compiled-controls']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor_smooth_compiled_exact_controls']
MODES['arb-tensor-fifteen-regions']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor_smooth_all_fifteen_original_regions']

MODES['arb-tensor-local53']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor_smooth_complete_local_region[53]']
MODES['arb-tensor-local63']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor_smooth_complete_local_region[63]']
MODES['arb-tensor-local65']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor_smooth_complete_local_region[65]']
MODES['arb-tensor-local67']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor_smooth_complete_local_region[67]']
MODES['arb-tensor-connection']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor_smooth_chart_connection_and_construction_deadline']
MODES['arb-tensor-complete-piece']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor_smooth_fresh_complete_original_piece']
MODES['arb-remaining-terms']=['tests/packaging/test_opaque_tight_transport.py::test_remaining_combined_terms_diagnostic']
MODES['arb-tangent-reuse-controls']=['tests/packaging/test_opaque_tight_transport.py::'+n for n in ('test_ordinary_exact_zero_work_preserves_all_mixed_endpoints','test_light_differential_reuses_only_current_scalar_box')]
MODES['arb-local-tensor-controls']=['tests/packaging/test_opaque_tight_transport.py::test_localized_tensor_contains_independent_exact_controls']
MODES['arb-local-tensor-regions']=['tests/packaging/test_opaque_tight_transport.py::test_localized_tensor_saved_smooth_regions']
MODES['arb-local-tensor-piece']=['tests/packaging/test_opaque_tight_transport.py::test_localized_tensor_fresh_complete_piece']
MODES['arb-tangent-reuse-cost']=['tests/packaging/test_opaque_tight_transport.py::test_redundant_work_controlled_cost_and_exact_piece_endpoints']
MODES['arb-gradient-correction-guidance']=['tests/packaging/test_opaque_tight_transport.py::test_gradient_correction_selects_axis_without_changing_acceptance']
MODES['arb-centered-tangent-controls']=['tests/packaging/test_opaque_tight_transport.py::'+n for n in ('test_tangent_centered_remainder_uses_zero_kernel_moment','test_tangent_centered_remainder_keeps_closed_log_motion')]
MODES['arb-centered-tangent-regions']=['tests/packaging/test_opaque_tight_transport.py::test_centered_tangent_original_saved_cells']
MODES['arb-direct-tangent-regions']=['tests/packaging/test_opaque_tight_transport.py::test_direct_tangent_range_saved_cells']
MODES['arb-fresh-schedule-controls']=['tests/packaging/test_opaque_tight_transport.py::'+n for n in ('test_fresh_refinement_schedule_checks_cover_and_charges_ancestors','test_fresh_refinement_schedule_recomputes_all_fields_and_remainders')]
MODES['arb-fresh-scheduled-piece']=['tests/packaging/test_opaque_tight_transport.py::test_fresh_scheduled_complete_piece']
MODES['arb-149-pretrial-controls']=MODES['arb-centered-tangent-controls']+MODES['arb-fresh-schedule-controls']+MODES['arb-147-driver-controls']+['tests/packaging/test_opaque_square_reference.py','tests/packaging/test_opaque_smooth_peano.py','tests/packaging/test_opaque_tangent_peano.py']
MODES['arb-full-light-controls']=['tests/packaging/test_opaque_tight_transport.py::test_full_light_differential_keeps_closed_mixed_coefficients']
MODES['arb-full-light-regions']=['tests/packaging/test_opaque_tight_transport.py::test_full_light_original_cells']
MODES['arb-compiled-inverse-controls']=['tests/packaging/test_opaque_tight_transport.py::'+n for n in ('test_compiled_inverse_keeps_independent_mixed_roots','test_compiled_inverse_keeps_complete_moving_measure')]
MODES['arb-compiled-inverse-regions']=['tests/packaging/test_opaque_tight_transport.py::test_compiled_original_cells']
MODES['arb-fourth-tangent-controls']=['tests/packaging/test_opaque_tight_transport.py::'+n for n in ('test_fourth_tangent_keeps_independent_log_motion','test_fourth_remainder_keeps_every_jump_order')]
MODES['arb-fourth-tangent-regions']=['tests/packaging/test_opaque_tight_transport.py::test_fourth_tangent_original_saved_cells']
MODES['arb-local4-controls']=['tests/packaging/test_opaque_tight_transport.py::test_quarter_localized_smooth_keeps_independent_controls']
MODES['arb-compiled-scheduled-piece']=['tests/packaging/test_opaque_tight_transport.py::test_compiled_scheduled_complete_piece']
MODES['arb-149-combined-controls']=MODES['arb-compiled-inverse-controls']+MODES['arb-fourth-tangent-controls']+MODES['arb-local4-controls']+MODES['arb-fresh-schedule-controls']+MODES['arb-147-driver-controls']+['tests/packaging/test_opaque_tight_transport.py::test_compiled_and_fourth_contract_preserves_refusal','tests/packaging/test_opaque_tight_transport.py::test_gradient_correction_selects_axis_without_changing_acceptance']
MODES['arb-conditioned-fourth-controls']=['tests/packaging/test_opaque_tight_transport.py::test_conditioned_fourth_keeps_independent_controls']
MODES['arb-conditioned-fourth-regions']=['tests/packaging/test_opaque_tight_transport.py::test_conditioned_fourth_original_saved_cells']
MODES['arb-conversion-cache-controls']=['tests/packaging/test_opaque_tight_transport.py::test_cached_arb_conversions_keep_exact_endpoints_and_context','tests/packaging/test_opaque_tight_transport.py::test_conditioned_correction_keeps_samples_domains_and_deadline']
MODES['arb-cached-conditioned-cost']=['tests/packaging/test_opaque_tight_transport.py::test_cached_conditioned_original_cost']
MODES['arb-weighted-quarter-controls']=['tests/packaging/test_opaque_tight_transport.py::test_localized_kernel_mass_encloses_independent_exact_moments','tests/packaging/test_opaque_tight_transport.py::test_weighted_quarter_smooth_keeps_independent_controls']
MODES['arb-range-selection']=['tests/packaging/test_opaque_tight_transport.py::test_all_scheduled_correction_ranges']
MODES['arb-range-select-controls']=['tests/packaging/test_opaque_tight_transport.py::test_conditioned_range_selection_retains_tiny_signed_enclosures']
MODES['arb-weighted-selection-controls']=MODES['arb-weighted-quarter-controls']+MODES['arb-range-select-controls']+MODES['arb-conditioned-fourth-controls']+MODES['arb-conversion-cache-controls']
MODES['arb-conditioned-scheduled-piece']=['tests/packaging/test_opaque_tight_transport.py::test_conditioned_scheduled_complete_piece']
MODES['arb-condition-escalation-controls']=['tests/packaging/test_opaque_tight_transport.py::test_conditioned_proof_reuses_samples_and_escalates_only_if_needed']
MODES['arb-escalation-qualified-controls']=MODES['arb-weighted-selection-controls']+MODES['arb-condition-escalation-controls']+MODES['arb-147-driver-controls']+MODES['arb-compiled-inverse-controls']
MODES['arb-contraction-controls']=['tests/packaging/test_opaque_tight_transport.py::test_certified_inverse_contraction_stop_keeps_closed_controls']
MODES['arb-contraction-cost']=['tests/packaging/test_opaque_tight_transport.py::test_inverse_contraction_compares_actual_whole_cells']
MODES['arb-calculation-profile']=['tests/packaging/test_opaque_tight_transport.py::test_remaining_complete_calculation_profile']
MODES['arb-fast-conversion-controls']=['tests/packaging/test_opaque_tight_transport.py::test_fast_arb_conversion_keeps_independent_formulas_and_tiny_terms']
MODES['arb-fast-conversion-cost']=['tests/packaging/test_opaque_tight_transport.py::test_fast_conversion_complete_local_cost']
MODES['arb-selected-driver-controls']=['tests/packaging/test_opaque_tight_transport.py::test_selected_candidate_contracts_and_construction_deadline']
MODES['arb-selected-piece']=['tests/packaging/test_opaque_tight_transport.py::test_selected_fresh_complete_piece']
MODES['arb-selected-fifteen']=['tests/packaging/test_opaque_tight_transport.py::test_selected_candidate_all_fifteen_smooth_regions']
MODES['arb-selected-combined-controls']=['tests/packaging/test_opaque_tight_transport.py::test_selected_proof_rules_in_fast_arithmetic']+MODES['arb-selected-driver-controls']
MODES['arb-reverse-conditioned-controls']=['tests/packaging/test_opaque_tight_transport.py::test_reverse_conditioned_identity_keeps_independent_controls']
MODES['arb-condition-directions']=['tests/packaging/test_opaque_tight_transport.py::test_both_conditioned_directions_on_remaining_regions']
MODES['arb-reverse-selection-controls']=MODES['arb-reverse-conditioned-controls']+MODES['arb-range-select-controls']+MODES['arb-condition-escalation-controls']
MODES['arb-repaired-selection-piece']=['tests/packaging/test_opaque_tight_transport.py::test_repaired_selection_scheduled_complete_piece']
MODES['arb-native-reuse-controls']=['tests/packaging/test_opaque_tight_transport.py::test_native_series_reuse_keeps_independent_controls_and_mutation']
MODES['arb-native-reuse-cost']=['tests/packaging/test_opaque_tight_transport.py::test_native_series_complete_local_cost']
MODES['arb-partitioned-range-controls']=['tests/packaging/test_opaque_tight_transport.py::test_partitioned_range_fallback_keeps_complete_measure_and_tiny_motion']
MODES['arb-partitioned-range-original']=['tests/packaging/test_opaque_tight_transport.py::test_partitioned_range_original_remaining_box']
MODES['arb-range-final-controls']=MODES['arb-selected-combined-controls']+MODES['arb-partitioned-range-controls']+MODES['arb-reverse-selection-controls']
MODES['arb-mixed-loss-controls']=['tests/packaging/test_opaque_tight_transport.py::test_mixed_loss_polynomial_contains_independent_coefficients','tests/packaging/test_opaque_tight_transport.py::test_mixed_loss_keeps_actual_segment_jumps_and_tiny_motion']
MODES['arb-mixed-local-cost']=['tests/packaging/test_opaque_tight_transport.py::test_mixed_material_actual_complete_local_cost']
MODES['arb-mixed-piece']=['tests/packaging/test_opaque_tight_transport.py::test_mixed_material_complete_piece']
MODES['arb-mixed-final-controls']=MODES['arb-mixed-loss-controls']+MODES['arb-range-final-controls']+['tests/packaging/test_opaque_tight_transport.py::test_mixed_material_contracts_and_construction_deadline']
MODES['arb-material-input-controls']=['tests/packaging/test_opaque_tight_transport.py::test_material_only_inputs_preserve_all_consumed_endpoints']
MODES['arb-material-input-cost']=['tests/packaging/test_opaque_tight_transport.py::test_material_only_actual_complete_local_cost']
MODES['arb-material-input-piece']=['tests/packaging/test_opaque_tight_transport.py::test_material_only_complete_piece']
MODES['arb-targeted-schedule-controls']=['tests/packaging/test_opaque_tight_transport.py::test_targeted_schedule_validates_complete_depth_and_charges']+MODES['arb-material-input-controls']
MODES['arb-targeted-material-piece']=['tests/packaging/test_opaque_tight_transport.py::test_targeted_material_complete_piece']
MODES['arb-direct-quarter-controls']=['tests/packaging/test_opaque_tight_transport.py::test_direct_quarter_conditioned_proof_keeps_independent_integrals']
MODES['arb-direct-quarter-cost']=['tests/packaging/test_opaque_tight_transport.py::test_direct_quarter_actual_complete_local_cost']
MODES['arb-front-correction-controls']=['tests/packaging/test_opaque_motion_extension.py::'+n for n in ['test_front_correction_selection_refuses_invalid_profiles','test_front_selected_correction_keeps_all_constructor_obligations']]
MODES['arb-front-correction-diagnostic']=['tests/packaging/test_opaque_motion_extension.py::test_front_saved_correction_comparison']
MODES['arb-front-schedule-controls']=['tests/packaging/test_opaque_motion_extension.py::test_front_schedule_forwards_complete_single_chart_and_charges','tests/packaging/test_opaque_tight_transport.py::test_fresh_refinement_schedule_checks_cover_and_charges_ancestors','tests/packaging/test_opaque_tight_transport.py::test_fresh_refinement_schedule_recomputes_all_fields_and_remainders']
MODES['arb-front-selected-reference']=['tests/packaging/test_opaque_motion_extension.py::test_front_selected_piece_reference']
MODES['arb-front-guidance-binding']=['tests/packaging/test_opaque_tangent_transverse.py::test_exact_partial_controls_crossing_count_before_localization','tests/packaging/test_opaque_motion_extension.py::test_front_selected_gradient_guidance_reaches_shared_integrator']
MODES['arb-front-parameter-diagnostic']=['tests/packaging/test_opaque_motion_extension.py::test_front_exact_partials_saved_boxes']
MODES['arb-front-material-controls']=['tests/packaging/test_opaque_motion_extension.py::test_front_material_only_keeps_all_consumed_endpoints']
MODES['arb-front-material-cost']=['tests/packaging/test_opaque_motion_extension.py::test_front_material_only_actual_cost']
MODES['arb-front-motion-controls']=['tests/packaging/test_opaque_tangent_transverse.py::test_exact_cosine_motion_keeps_intrinsic_and_coordinate_chain']
MODES['arb-front-motion-cost']=['tests/packaging/test_opaque_motion_extension.py::test_front_exact_motion_actual_cost']
MODES['arb-front-axis-binding']=['tests/packaging/test_opaque_motion_extension.py::test_front_axis_selection_requires_conditioned_profile']
MODES['arb-front-guided-reference']=['tests/packaging/test_opaque_motion_extension.py::test_front_guided_piece_reference']
MODES['arb-front-directions']=['tests/packaging/test_opaque_motion_extension.py::test_front_conditioned_directions_saved_comparison']
MODES['arb-front-partial-binding']=['tests/packaging/test_opaque_tangent_transverse.py::test_exact_partial_refinement_activates_uncertain_crossings_and_refuses_disjoint_proof']
MODES['arb-tangent-transverse-controls']=['tests/packaging/test_opaque_tangent_transverse.py']
MODES['arb-front-transverse-controls']=['tests/packaging/test_opaque_motion_extension.py::test_front_transverse_profile_requires_the_qualified_correction','tests/packaging/test_opaque_motion_extension.py::test_front_transverse_constructor_preserves_selected_profile']
MODES['arb-front-transverse-diagnostic']=['tests/packaging/test_opaque_motion_extension.py::test_front_transverse_saved_comparison']
MODES['arb-front-proof-profile']=['tests/packaging/test_opaque_motion_extension.py::test_front_remaining_proof_components']
MODES['arb-front-partial-controls']=['tests/packaging/test_opaque_tangent_transverse.py::test_exact_cosine_partials_enclose_independent_closed_derivatives']
MODES['arb-front-final-controls']=MODES['arb-tangent-transverse-controls']+MODES['arb-front-axis-binding']+MODES['arb-front-guidance-binding']+MODES['arb-front-transverse-controls']+MODES['arb-front-correction-controls']+MODES['arb-front-schedule-controls']+['tests/packaging/test_opaque_tight_transport.py::test_fourth_tangent_keeps_independent_log_motion','tests/packaging/test_opaque_tight_transport.py::test_gradient_correction_selects_axis_without_changing_acceptance']
MODES['arb-front-material-final-controls']=MODES['arb-front-final-controls']+MODES['arb-front-material-controls']
MODES['face-certificate-controls']=['tests/packaging/test_opaque_face_certificate.py']
MODES['arb-motion-selection-controls']=['tests/packaging/test_opaque_motion_extension.py::'+n for n in ['test_requested_motion_reaches_geometry_before_construction','test_angular_selection_refuses_undeclared_inputs_before_geometry','test_complete_and_failed_reports_keep_requested_identity']]
MODES['arb-motion-extension-controls']=['tests/packaging/test_opaque_motion_extension.py::'+n for n in ['test_declared_motion_builds_fresh_complete_cover_and_keeps_horizon','test_selected_arithmetic_retains_independent_roots_measure_and_tiny_motion','test_selected_profile_refuses_a_noncurved_piece']]
MODES['arb-motion-extension-reference']=['tests/packaging/test_opaque_motion_extension.py::test_selected_motion_piece_fresh_reference']
MODES['arb-final-selected-controls']=MODES['arb-mixed-final-controls']+MODES['arb-targeted-schedule-controls']+MODES['arb-direct-quarter-controls']
MODES['arb-boundary-flux-diagnostic']=['tests/packaging/test_opaque_boundary_flux.py::test_saved_boundary_components']
MODES['arb-boundary-weighted-diagnostic']=['tests/packaging/test_opaque_boundary_flux.py::test_saved_weighted_jump_components']
MODES['arb-boundary-mass-controls']=['tests/packaging/test_opaque_boundary_flux.py::'+n for n in ['test_weighted_mass_uses_exact_first_segment_and_retains_tiny_terms','test_weighted_mass_contains_independent_closed_polynomial','test_weighted_mass_keeps_uncertain_roots_and_refuses_invalid_support']]
MODES['arb-boundary-mass-controls']+=['tests/packaging/test_opaque_boundary_flux.py::test_weighted_mass_selected_arithmetic_and_bounded_cache']
MODES['arb-boundary-mass-comparison']=['tests/packaging/test_opaque_boundary_flux.py::test_saved_weighted_mass_comparison']
MODES['arb-boundary-mass-final-controls']=MODES['arb-boundary-mass-controls']+MODES['arb-front-material-final-controls']
MODES['arb-boundary-cost-profile']=['tests/packaging/test_opaque_boundary_flux.py::test_saved_boundary_cost_profile']
MODES['arb-boundary-line-controls']=['tests/packaging/test_opaque_boundary_flux.py::test_quadrature_lines_skip_unused_continuous_proofs','tests/packaging/test_opaque_boundary_flux.py::test_actual_front_builds_continuous_proofs_only_on_consumed_strips']
MODES['arb-boundary-line-comparison']=['tests/packaging/test_opaque_boundary_flux.py::test_saved_line_proof_endpoints_and_cost']
MODES['arb-boundary-final-controls']=MODES['arb-boundary-line-controls']+MODES['arb-boundary-mass-final-controls']
MODES['arb-motion-progress-directions']=['tests/packaging/test_opaque_motion_progress.py::test_remaining_front_piece_proof_directions']

MODES['arb-motion-root-controls']=['tests/packaging/test_opaque_motion_progress.py::test_crossing_contraction_contains_all_independent_nonlinear_roots','tests/packaging/test_opaque_motion_progress.py::test_crossing_contraction_keeps_tiny_roots_uncertainty_and_cancellation']

MODES['arb-motion-root-binding']=['tests/packaging/test_opaque_motion_progress.py::test_discrete_line_proof_contracts_before_forming_jumps','tests/packaging/test_opaque_motion_progress.py::test_crossing_contraction_selected_backend']

MODES['arb-motion-edge-binding']=['tests/packaging/test_opaque_motion_progress.py::test_continuous_proof_contracts_only_certified_interior_edge_roots']

MODES['arb-motion-root-final']=MODES['arb-motion-root-controls']+MODES['arb-motion-root-binding']+MODES['arb-motion-edge-binding']+MODES['arb-boundary-final-controls']

MODES['arb-motion-ratio-controls']=['tests/packaging/test_opaque_motion_progress.py::test_exact_slope_ratio_cancels_shared_norm_and_retains_full_box','tests/packaging/test_opaque_motion_progress.py::test_transverse_proof_uses_independent_exact_ratio']

MODES['arb-motion-progress-components']=['tests/packaging/test_opaque_motion_progress.py::test_remaining_front_piece_bound_components']

MODES['arb-motion-coordinate-controls']=['tests/packaging/test_opaque_motion_progress.py::test_remaining_front_cyclic_maps_preserve_physical_fields_and_motion','tests/packaging/test_opaque_motion_progress.py::test_remaining_front_coordinate_feasibility']

MODES['arb-motion-front-final']=MODES['arb-motion-root-final']+MODES['arb-motion-ratio-controls']+['tests/packaging/test_opaque_motion_progress.py::test_remaining_front_cyclic_maps_preserve_physical_fields_and_motion']

MODES['arb-motion-progress-subdivision']=['tests/packaging/test_opaque_motion_progress.py::test_remaining_front_subdivision_response']

MODES['arb-motion-root-cost-control']=['tests/packaging/test_opaque_motion_progress.py::test_contraction_reuses_independent_slope_without_rebuilding_field_jets','tests/packaging/test_opaque_motion_progress.py::test_independent_slope_reuse_keeps_uncertain_roots_and_full_bracket']

MODES['arb-next-piece-transport-cost']=['tests/packaging/test_opaque_motion_progress.py::test_next_piece_saved_transport_and_contraction_cost']

MODES['arb-next-piece-subdivision']=['tests/packaging/test_opaque_motion_progress.py::test_next_piece_subdivision_response']

MODES['face-bundle-controls']=['tests/packaging/test_opaque_face_certificate.py::test_face_bundle_checks_input_artifact_bytes_and_recomputed_sum']

MODES['original-face-motion-installed']=['tests/python/test_opaque_piece_motion_acceptance.py::test_original_motion_meets_independent_complete_face']
MODES['opaque-cuda-original-face-motion-installed']=MODES['original-face-motion-installed']

MODES['face-composition-final-controls']=['tests/packaging/test_opaque_face_certificate.py']

MODES['flat-refinement-controls']=['tests/packaging/test_opaque_flat_resample.py']

MODES['flat-face-tensor-diagnostic']=['tests/python/test_opaque_piece_motion_acceptance.py::test_complete_face_flat_tensor_diagnostic']

MODES['flat-native-controls']=['tests/python/test_native_opaque_flat.py::'+n for n in ('test_flat_capacity_keeps_closed_moments_and_area_motion','test_flat_refinement_preserves_curved_priority_and_combined_capacity','test_flat_refinement_retains_independent_face_and_logical_block_refusals')]
MODES['opaque-cuda-flat-native-controls']=['tests/python/test_native_opaque_flat.py']

MODES['ptx-resource-controls']=['tests/packaging/test_ptx_resource.py','tests/packaging/test_ptx_embedding.py']

MODES['ordered-motion-controls']=['tests/packaging/test_opaque_streamed_motion.py','tests/python/test_shading_reference_channels.py']


MODES['arb-piece10-directions']=['tests/packaging/test_opaque_motion_progress.py::test_piece10_saved_correction_directions_and_refinement']
MODES['arb-piece10-map-controls']=['tests/packaging/test_opaque_motion_progress.py::test_remaining_front_cyclic_maps_preserve_physical_fields_and_motion']


MODES['arb-remaining-curved-map-controls']=['tests/packaging/test_opaque_motion_tangent.py::test_original_curved_map_keeps_moving_horizon_zero']
MODES['arb-remaining-outside-reference']=['tests/packaging/test_opaque_motion_extension.py::test_remaining_outside_curved_fresh_reference']


MODES['arb-piece9-transverse-diagnostic']=['tests/packaging/test_opaque_motion_extension.py::test_piece9_saved_transverse_correction']


MODES['arb-curved-transverse-controls']=['tests/packaging/test_opaque_tangent_transverse.py::test_curved_crossing_transverse_integral_has_closed_polynomial_reference','tests/packaging/test_opaque_tangent_transverse.py::test_curved_transverse_profile_is_explicit_and_preserves_monotonicity_refusal']

MODES['arb-curved-transverse-binding-controls']=['tests/packaging/test_opaque_motion_extension.py::test_curved_transverse_profile_refuses_unqualified_cover_before_geometry','tests/packaging/test_opaque_motion_extension.py::test_curved_transverse_whole_reference_keeps_profile_identity','tests/packaging/test_opaque_motion_extension.py::test_curved_transverse_real_cover_forwards_each_chart_and_keeps_motion']

MODES['arb-curved-proof-cost-controls']=['tests/packaging/test_opaque_tangent_transverse.py::test_curved_quadrature_lines_keep_crossings_without_optional_newton_work','tests/packaging/test_opaque_tangent_transverse.py::test_transverse_empty_jump_list_reuses_complete_smooth_proof_without_field_work']

MODES['arb-inside-motion-reference']=['tests/packaging/test_opaque_motion_extension.py::test_remaining_inside_curved_fresh_reference']

MODES['arb-inside-angle-controls']=['tests/packaging/test_opaque_inside_angle.py','tests/packaging/test_opaque_algebraic_angle_inverse.py::test_unsquared_inverse_matches_closed_mixed_coefficients']

MODES['arb-inside-angular-reference']=['tests/packaging/test_opaque_motion_extension.py::test_inside_angular_piece_fresh_reference']

MODES['arb-front-tolerance-controls']=['tests/packaging/test_opaque_motion_extension.py::test_stronger_front_reference_tolerance_rejects_looser_or_invalid_values','tests/packaging/test_opaque_motion_extension.py::test_stronger_front_reference_tolerance_is_forwarded_without_changing_limits']

MODES['pixel-certificate-controls']=['tests/packaging/test_opaque_face_certificate.py']

MODES['pixel-motion-harness-controls']=['tests/packaging/test_opaque_streamed_motion.py']

MODES['original-pixel-motion-installed']=['tests/python/test_opaque_piece_motion_acceptance.py::test_original_motion_meets_independent_complete_pixel']
MODES['opaque-cuda-original-pixel-motion-installed']=MODES['original-pixel-motion-installed']

MODES['arb-failed-normal-regions']=['tests/packaging/test_opaque_motion_extension.py::test_failed_normal_saved_refinement_directions']

MODES['arb-remaining-front-regions']=['tests/packaging/test_opaque_motion_extension.py::test_remaining_front_saved_refinement_directions']

MODES['arb-front-fixed-refinement-controls']=['tests/packaging/test_opaque_motion_extension.py::test_front_fixed_refinement_axis_refuses_invalid_or_unqualified_selection','tests/packaging/test_opaque_motion_extension.py::test_front_fixed_refinement_axis_is_geometry_only','tests/packaging/test_opaque_motion_extension.py::test_front_fixed_refinement_axis_refuses_a_curved_chart']

MODES['arb-polynomial-input-controls']=['tests/packaging/test_opaque_polynomial_inputs.py::test_rotating_normal_has_closed_full_taylor_and_tiny_motion','tests/packaging/test_opaque_polynomial_inputs.py::test_moving_view_has_closed_full_taylor','tests/packaging/test_opaque_polynomial_inputs.py::test_coordinate_motion_retains_all_mixed_terms_after_norm_cancellation','tests/packaging/test_opaque_polynomial_inputs.py::test_polynomial_inputs_refuse_invalid_norms_or_jets','tests/packaging/test_opaque_polynomial_inputs.py::test_original_front_inputs_keep_all_coefficients_and_jacobian','tests/packaging/test_opaque_polynomial_inputs.py::test_polynomial_reference_rejects_invalid_or_unqualified_profile','tests/packaging/test_opaque_polynomial_inputs.py::test_polynomial_reference_uses_full_model_and_original_jacobian','tests/packaging/test_opaque_polynomial_inputs.py::test_polynomial_reference_refuses_unqualified_curved_mapping']

MODES['arb-polynomial-input-diagnostic']=['tests/packaging/test_opaque_polynomial_inputs.py::test_saved_front_polynomial_input_cost_and_bounds']

MODES['arb-curved-tolerance-controls']=['tests/packaging/test_opaque_curved_tolerance.py']

MODES['arb-polynomial-light-controls']=['tests/packaging/test_opaque_polynomial_inputs.py::test_light_only_keeps_closed_coordinate_chain_without_unused_view_work','tests/packaging/test_opaque_polynomial_inputs.py::test_light_highlight_rejects_invalid_inputs','tests/packaging/test_opaque_polynomial_inputs.py::test_physical_light_highlight_keeps_original_full_coefficients']

MODES['arb-polynomial-light-binding']=['tests/packaging/test_opaque_polynomial_light_binding.py::test_polynomial_light_refuses_unqualified_outside_before_geometry','tests/packaging/test_opaque_polynomial_light_binding.py::test_polynomial_light_complete_and_failed_identity','tests/packaging/test_opaque_polynomial_light_binding.py::test_polynomial_light_physical_callbacks_keep_angle_measure_and_mixed_motion']

MODES['arb-polynomial-light-diagnostic']=['tests/packaging/test_opaque_polynomial_light_binding.py::test_saved_curved_polynomial_light_cost_and_complete_bound_components']

MODES['arb-horizon-material-query-controls']=['tests/packaging/test_opaque_polynomial_light_binding.py::test_horizon_material_query_forwards_only_consumed_fields']

MODES['arb-checker-cost']=['tests/packaging/test_opaque_checker_cost.py::test_saved_piece8_cost']

MODES['arb-chart-cache-controls']=['tests/packaging/test_opaque_checker_cost.py::test_chart_cache_keeps_unclipped_values_and_fresh_local_bounds']

MODES['arb-physical-slope-controls']=['tests/packaging/test_opaque_checker_cost.py::test_cancelled_physical_slope_keeps_closed_mixed_coefficients','tests/packaging/test_opaque_checker_cost.py::test_cancelled_physical_slope_removes_interval_self_dependency']

MODES['arb-slope-binding-controls']=['tests/packaging/test_opaque_tight_transport.py::test_compiled_inverse_keeps_independent_mixed_roots', 'tests/packaging/test_opaque_tight_transport.py::test_compiled_inverse_keeps_complete_moving_measure', 'tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_original_map_matches_complete_material_jet', 'tests/packaging/test_opaque_algebraic_angle_inverse.py::test_physical_selector_and_construction_deadline', 'tests/packaging/test_opaque_polynomial_light_binding.py::test_polynomial_light_physical_callbacks_keep_angle_measure_and_mixed_motion']

MODES['arb-shared-slope-controls']=['tests/packaging/test_opaque_checker_cost.py::test_shared_inverse_slope_keeps_closed_root_measure_and_mixed_motion','tests/packaging/test_opaque_checker_cost.py::test_shared_inverse_slope_rejects_unqualified_query']

MODES['original-public-motion-installed']=['tests/python/test_opaque_public_motion.py::test_public_original_motion_against_complete_reference']
MODES['opaque-cuda-original-public-motion-installed']=MODES['original-public-motion-installed']

MODES['arb-projected-motion-controls']=['tests/packaging/test_opaque_projected_motion.py::'+name for name in ['test_moving_projected_basis_has_exact_vertex_identities', 'test_moving_projected_fields_keep_closed_perspective_and_horizon_motion', 'test_moving_projected_fields_refuse_invalid_geometry', 'test_aligned_world_motion_has_closed_projection_and_view_chain', 'test_affine_shear_recomputed_area_normals_follow_exact_cofactor', 'test_original_world_prepare_keeps_base_fields_and_complete_motion', 'test_world_reference_forwards_explicit_profile_without_relabeling']]
MODES['arb-original-world-reference']=['tests/packaging/test_opaque_projected_motion.py::test_original_world_fresh_reference']

MODES['original-world-profile']=['tests/python/test_opaque_public_motion.py::test_extract_original_world_motion_profile']

MODES['arb-world-physical-controls']=['tests/packaging/test_opaque_projected_motion.py::test_original_world_physical_map_keeps_implicit_angle_and_complete_measure']

MODES['arb-quadratic-dual-controls']=['tests/packaging/test_opaque_projected_motion.py::test_quadratic_first_motion_product_does_not_construct_discarded_second_motion','tests/packaging/test_opaque_projected_motion.py::test_quadratic_first_motion_product_retains_complexity_guard_for_consumed_terms']

MODES['arb-quadratic-scale-controls']=['tests/packaging/test_opaque_projected_motion.py::test_normalized_quadratic_root_preserves_closed_large_rational_scale','tests/packaging/test_opaque_projected_motion.py::test_normalized_quadratic_root_refuses_ambiguous_or_nonpositive_scale']

MODES["arb-quadratic-legacy-controls"]=["tests/packaging/test_opaque_knot_partition.py::"+name for name in ("test_quadratic_sqrt_keeps_exact_square_and_moving_composition","test_quadratic_sqrt_nonlinear_mixed_matches_point_algebra","test_quadratic_sqrt_refuses_nonpositive_domain_and_keeps_tiny_curvature")]
MODES['arb-world-normal-controls']=['tests/packaging/test_opaque_projected_motion.py::test_world_shear_normalization_precedes_perspective_interpolation']
MODES['arb-curved-sixth-controls']=['tests/packaging/test_opaque_projected_motion.py::test_curved_sixth_lines_keep_closed_log_integrals','tests/packaging/test_opaque_projected_motion.py::test_curved_sixth_lines_keep_nonlinear_change_of_variable']
MODES['arb-curved-root-controls']=['tests/packaging/test_opaque_projected_motion.py::test_curved_partitioned_roots_keep_closed_nonlinear_integral','tests/packaging/test_opaque_projected_motion.py::test_root_partition_requires_qualified_fixed_count','tests/packaging/test_opaque_tangent_transverse.py::test_curved_transverse_profile_is_explicit_and_preserves_monotonicity_refusal']
MODES['arb-radial-local-controls']=['tests/packaging/test_opaque_tensor_smooth.py::test_radial_quarter_peano_masses_enclose_independent_antiderivative','tests/packaging/test_opaque_tensor_smooth.py::test_biaxis_local_tensor_preserves_closed_integrals_and_all_contributions']
MODES['arb-pure-flux-controls']=['tests/packaging/test_opaque_projected_motion.py::test_explicit_pure_flux_reuse_keeps_every_endpoint_and_tiny_term','tests/packaging/test_opaque_projected_motion.py::test_pure_flux_reuse_refuses_implicit_contract']
MODES['arb-curved-root-binding']=['tests/packaging/test_opaque_projected_motion.py::test_curved_root_profile_reaches_full_checker','tests/packaging/test_opaque_projected_motion.py::test_curved_root_profile_rejects_unqualified_selection']
MODES['arb-world-cost']=['tests/packaging/test_opaque_checker_cost.py::test_world_saved_piece8_cost']
MODES['arb-world-axis-controls']=['tests/packaging/test_opaque_projected_motion.py::test_world_angular_correction_direction_reaches_every_chart','tests/packaging/test_opaque_projected_motion.py::test_angular_correction_direction_refuses_invalid_selection']
MODES['arb-world-direction-integrals']=['tests/packaging/test_opaque_tangent_transverse.py::test_transverse_tangent_preserves_closed_integrals']
MODES['arb-world-local-crossings']=['tests/packaging/test_opaque_projected_motion.py::test_local_crossing_range_excludes_only_provably_absent_knot','tests/packaging/test_opaque_projected_motion.py::test_local_crossing_range_requires_explicit_boolean']
MODES['arb-world-local-integrals']=['tests/packaging/test_opaque_projected_motion.py::test_local_crossing_proof_retains_closed_piecewise_integrals']
MODES['arb-world-front-cost']=['tests/packaging/test_opaque_checker_cost.py::test_world_front_saved_corrections']
MODES['arb-world-sixth-line-controls']=['tests/packaging/test_opaque_projected_motion.py::'+n for n in ['test_sixth_line_proof_keeps_closed_log_integrals','test_mixed_fourth_sixth_rule_keeps_all_closed_jumps','test_sixth_line_proof_rejects_unqualified_profiles']]
MODES['arb-rgb-transport-reuse']=['tests/packaging/test_opaque_projected_motion.py::test_identical_rgb_transport_reuses_only_equal_endpoint_proofs']
MODES['arb-front-line-binding']=['tests/packaging/test_opaque_projected_motion.py::test_front_line_order_reaches_qualified_correction','tests/packaging/test_opaque_projected_motion.py::test_front_line_order_refuses_invalid_selection']
MODES['arb-world-cancelled-motion']=['tests/packaging/test_opaque_projected_motion.py::test_cancelled_polynomial_motion_preserves_complete_coordinate_chain','tests/packaging/test_opaque_projected_motion.py::test_cancelled_polynomial_motion_requires_explicit_boolean']
MODES['arb-world-crossing-support']=['tests/packaging/test_opaque_projected_motion.py::'+n for n in ['test_crossing_support_keeps_every_possible_transverse_root','test_crossing_support_preserves_closed_log_integrals','test_crossing_support_requires_explicit_boolean']]
MODES['arb-front-support-binding']=['tests/packaging/test_opaque_projected_motion.py::test_front_crossing_support_reaches_qualified_correction','tests/packaging/test_opaque_projected_motion.py::test_front_crossing_support_refuses_invalid_selection']
MODES['arb-front-material-cache']=['tests/packaging/test_opaque_projected_motion.py::test_front_material_cache_preserves_fresh_region_bounds']
MODES['arb-world-root-partition']=['tests/packaging/test_opaque_projected_motion.py::'+n for n in ['test_partitioned_crossing_keeps_closed_two_axis_log_integral','test_root_partition_requires_qualified_fixed_count']]

MODES['arb-tensor-path-audit']=['tests/packaging/test_opaque_checker_cost.py::test_saved_tensor_proof_path_audit']
MODES['arb-tensor-requalify']=['tests/packaging/test_opaque_projected_motion.py::test_requalify_saved_tensor_certificate']
MODES['arb-tensor-identity-regression']=sum((MODES[k] for k in ('arb-world-direction-integrals','arb-world-local-integrals','arb-world-sixth-line-controls','arb-world-crossing-support')),[])
MODES['arb-polynomial-prepare-work']=['tests/packaging/test_opaque_projected_motion.py::'+n for n in ('test_polynomial_preparation_omits_only_unconsumed_scalar_work','test_polynomial_prepare_rebuilds_on_context_changes_without_evaluating')]
MODES['arb-weighted-kernel-reuse']=['tests/packaging/test_opaque_projected_motion.py::test_weighted_kernel_reuse_preserves_all_frozen_endpoints']
MODES['arb-weighted-context']=['tests/packaging/test_opaque_projected_motion.py::test_weighted_kernel_cache_respects_context_and_capacity']
MODES['arb-front-root-binding']=['tests/packaging/test_opaque_projected_motion.py::'+n for n in ('test_front_root_partition_reaches_qualified_correction','test_front_root_partition_refuses_unqualified_profile')]
MODES['arb-tensor-cost-regression']=sum((MODES[k] for k in ('arb-front-root-binding','arb-weighted-context','arb-polynomial-input-controls','arb-polynomial-light-controls','arb-tensor-identity-regression','arb-world-root-partition')),[])
MODES['arb-world-merge-domains']=['tests/packaging/test_opaque_checker_cost.py::test_world_front_merge_domain_preflight']
MODES['arb-front-sample-reuse']=['tests/packaging/test_opaque_projected_motion.py::'+n for n in ('test_front_full_samples_share_only_identical_raw_material_fields','test_material_seed_freezes_fields_and_rejects_conflicting_values')]
MODES['worker-identity-controls']=['tests/packaging/test_opaque_worker_identity.py']
MODES['camera-reversal-controls']=['tests/packaging/test_opaque_camera_reversal.py']
MODES['physical-pixel-diagnostic-installed']=['tests/python/test_opaque_piece_motion_acceptance.py::test_complete_physical_pixel_diagnostic']
MODES['flat-reallocation-controls-installed']=['tests/packaging/test_opaque_flat_resample.py']
MODES['physical-flat-diagnostic-installed']=['tests/python/test_opaque_piece_motion_acceptance.py::test_physical_flat_reallocation_diagnostic']
MODES['symmetric-triangle-controls-installed']=['tests/packaging/test_opaque_symmetric_triangle.py']
MODES['physical-symmetric-diagnostic-installed']=['tests/python/test_opaque_piece_motion_acceptance.py::test_physical_symmetric_triangle_diagnostic']
MODES['flat-transport-controls-installed']=['tests/packaging/test_opaque_flat_transport.py']
MODES['physical-flat-transport-installed']=['tests/python/test_opaque_piece_motion_acceptance.py::test_physical_flat_transport_diagnostic']
MODES['material-transport-controls-installed']=['tests/python/test_native_opaque_transport.py']
MODES['opaque-cuda-material-transport-controls-installed']=MODES['material-transport-controls-installed']
MODES['opaque-cuda-material-raw-regression-installed']=['tests/python/test_native_opaque_flat.py']
MODES['transport-rebalanced-diagnostic-installed']=['tests/python/test_opaque_piece_motion_acceptance.py::test_transport_rebalanced_pixel_diagnostic']
MODES['scene-horizon-qualified-installed']=MODES['scene-horizon']
MODES['opaque-cuda-scene-horizon-qualified-installed']=MODES['scene-horizon']
MODES['material-linear-cut-installed']=['tests/python/test_native_opaque_transport.py::test_material_transport_linear_cut_retains_one_sided_interior_flux']
MODES['transport-teaching-controls']=['tests/python/test_testing_dashboard.py::test_every_repository_test_has_an_authored_explanation','tests/python/test_testing_dashboard.py::test_explanations_are_read_only_guides_not_historical_test_results','tests/python/test_testing_dashboard.py::test_named_check_explanations_describe_the_comparison_and_its_limits']
MODES['arb-polynomial-row-reuse']=['tests/packaging/test_opaque_polynomial_inputs.py::test_polynomial_row_reuse_keeps_frozen_endpoints_and_private_values','tests/packaging/test_opaque_polynomial_inputs.py::test_polynomial_row_reuse_bounds_storage_and_rebuilds_after_context_change']
MODES['arb-polynomial-zero-horner']=['tests/packaging/test_opaque_polynomial_inputs.py::test_polynomial_horner_omits_only_leading_exact_zero_products']
MODES['arb-shear-range-cover']=['tests/packaging/test_opaque_checker_cost.py::test_saved_shear_cover_natural_range_cost']
MODES['arb-small-range-selection']=['tests/packaging/test_opaque_tangent_peano.py::test_conditioned_range_selection_retains_complete_small_integral']
MODES['arb-root-refinement-selection']=['tests/packaging/test_opaque_tangent_transverse.py::test_partitioned_transverse_skips_only_already_enclosed_small_jump']
MODES['arb-angle-scalar-reuse']=['tests/packaging/test_opaque_polynomial_inputs.py::test_angle_differential_reuses_only_identical_scalar_domains']
MODES['arb-curved-sample-reuse']=['tests/packaging/test_opaque_checker_cost.py::test_curved_full_samples_seed_identical_raw_material_fields']
MODES['arb-root-contraction-work']=['tests/packaging/test_opaque_tangent_transverse.py::test_crossing_contraction_stops_after_sufficient_enclosed_reduction']
MODES['arb-shared-add-context']=['tests/packaging/test_opaque_polynomial_inputs.py::test_jet_addition_shares_context_without_changing_frozen_endpoints']
MODES['arb-tensor36-regions']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor36_saved_shear_regions_combined_bounds']
MODES['arb-tensor36-selection']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor36_chart_selection_is_explicit_and_bound']
MODES['arb-scalar-inverse-controls']=['tests/packaging/test_opaque_checker_cost.py::test_scalar_implicit_inverse_preserves_closed_motion_without_series']
MODES['arb-tensor48-controls']=['tests/packaging/test_opaque_tensor_smooth.py::test_tensor48_radial_moments_full_integrals_and_work']
MODES['arb-scalar-horner-controls']=['tests/packaging/test_opaque_polynomial_inputs.py::test_prepared_scalar_horner_shares_context_and_keeps_frozen_endpoints']
MODES['arb-tensor36-controls']=['tests/packaging/test_opaque_tensor_smooth.py::test_gauss12_positive_rule_independent_moments_and_kernels','tests/packaging/test_opaque_tensor_smooth.py::test_tensor36_complete_integrals_corrections_and_work_charge']
MODES['arb-tensor48-regression']=sum((MODES[k] for k in ('arb-tensor-cost-regression','arb-tensor36-controls','arb-tensor36-selection','arb-tensor48-controls','arb-shared-slope-controls','arb-scalar-inverse-controls')),[])
if __name__=='__main__':raise SystemExit(main())
