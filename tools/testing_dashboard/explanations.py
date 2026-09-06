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
cam17_cuda_gradcheck_on_device|Check camera differentiation while the computation runs on a GPU.|Run the focal-length numerical gradient fixture on CUDA when the required device is available.
cam18_invalid_construction_is_rejected|Reject camera settings that violate the public contract.|Try the parameterized invalid-construction case and require the documented exception.
cam19_unproject_of_an_infinite_background_is_zero_valid_false_and_free_of_nan|Handle empty depth-image background without poisoning coordinates or gradients.|Unproject infinite background depths; require invalid flags, zero background coordinates and gradients, and no NaNs.
cam20_the_facade_computes_nothing|Keep the convenient camera API consistent with the framework-specific API.|Run equivalent calls through gffx.camera and gffx.torch.camera and require exact output equality.
""")
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


def explain(nodeid, check=None):
    """Unknown identifiers are data, never filesystem paths or inferred evidence."""
    key=nodeid.split('[',1)[0]
    value=GUIDES.get(key) if check is None else CHECK_GUIDES.get((key,check))
    return dict(authored=value is not None, basis='current author guide', **(value or dict(
        purpose='No authored explanation is available for this test or named check yet.',
        method='Use the source identifier and reported diagnostics to inspect it; the dashboard does not infer intent from a hash or assertion result.',
        limits='An unexplained test result is not a claim of feature acceptance. Add a reviewed guide before treating this as a maintained inspection page.')))
