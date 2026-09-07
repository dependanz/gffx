"""Reviewed reader guides, separate from execution evidence. No source-path reads.

Each row is test name | purpose | method. Parameter variants share the guide,
not their result or identity. The repository inventory test enforces new entries.
"""
GUIDES = {}
CHECK_GUIDES = {}


def _add(path, limits, rows):
    for row in rows.strip().splitlines():
        name, purpose, method = row.split('|')
        key = 'tests/' + path + '.py::test_' + name
        if key in GUIDES: raise ValueError('duplicate test guide: ' + key)
        GUIDES[key] = dict(purpose=purpose, method=method, limits=limits)


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

def explain(nodeid, check=None):
    """Unknown identifiers are data, never filesystem paths or inferred evidence."""
    key=nodeid.split('[',1)[0]
    value=GUIDES.get(key) if check is None else CHECK_GUIDES.get((key,check))
    return dict(authored=value is not None, basis='current author guide', **(value or dict(
        purpose='No authored explanation is available for this test or named check yet.',
        method='Use the source identifier and reported diagnostics to inspect it; the dashboard does not infer intent from a hash or assertion result.',
        limits='An unexplained test result is not a claim of feature acceptance. Add a reviewed guide before treating this as a maintained inspection page.')))
