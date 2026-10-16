"""Small fixtures and arithmetic forecasts only; no large benchmark entry point."""
from fractions import Fraction as Q
import json
import math
import struct
import sys

PROPOSED_ROWS = 16_777_216
PROPOSED_RATIONAL_BITS = 6144
MAX_FOCUSED_ROWS = 4096


def make_fixture(rows, bits=PROPOSED_RATIONAL_BITS):
    if type(rows) is not int or not 0 <= rows <= MAX_FOCUSED_ROWS:
        raise ValueError("focused fixture limit; large runs require a new reviewed harness")
    if type(bits) is not int or not 256 <= bits <= 8192:
        raise ValueError("focused exact rational bit range")
    shared = {"schema": "history-shaped serializer stress, not scientific evidence",
              "bounds": (Q(1, 10**12), Q(1, 10**10))}
    traces = [(j, Q((1 << bits) + j, 2*j + 3), Q(-j, 2*j + 5),
               {"shared": shared, "state": (True, None, "é")})
              for j in range(rows)]
    return {"coordinator_history": tuple(traces), "pixel_traces": traces,
            "texture": shared, "comparison_ordinary_history": ((0, True, None),),
            "comparison_array_shapes": {"colour": (1, 32, 32, 3)},
            "comparison_array_sha256": {"colour": "0" * 64}}


def reference_dump(value, out):
    # Independent spelling of the frozen codec specification. Used only for
    # trusted fixtures; it is never a production path that omits preflight.
    def default(item):
        if type(item) is not Q:
            raise TypeError("ordinary exact Fraction only")
        return {"rational_hex": [hex(item.numerator).replace("0x", ""),
                                 hex(item.denominator).replace("0x", "")]}
    json.dump(value, out, default=default, separators=(",", ":"),
              ensure_ascii=True, allow_nan=False)


def retained_shallow_sum(value):
    """Small-sample object sum, including Fraction internals and dictionary keys."""
    seen = set()
    pending = [value]
    size = 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if type(item) is Q:
            pending.extend((item.numerator, item.denominator))
        elif type(item) is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
    return size


def resource_forecast():
    # Two tiny shape samples; no large graph, serialization or stress run.
    from disk_exact_json import IndexLimits, validate_exact_graph
    small = make_fixture(512)
    larger = make_fixture(1024)
    settings = IndexLimits(cache_kib=256)
    first = validate_exact_graph(small, limits=settings)
    second = validate_exact_graph(larger, limits=settings)
    node_slope = (second["validated_unique_objects"] -
                  first["validated_unique_objects"]) // 512
    edge_slope = (second["validated_container_edges"] -
                  first["validated_container_edges"]) // 512
    assert node_slope == 5 and edge_slope == 8
    nodes = node_slope * PROPOSED_ROWS + first["validated_unique_objects"] - 5*512
    edges = edge_slope * PROPOSED_ROWS + first["validated_container_edges"] - 8*512
    first_size = retained_shallow_sum(small)
    second_size = retained_shallow_sum(larger)
    # Assumption: proportional layout. Use the larger sample ratio and add
    # 25% for allocator slack/fragmentation; this is an estimate, not a bound.
    graph_estimate = math.ceil(PROPOSED_ROWS * max(first_size/512, second_size/1024))
    pages = math.ceil(nodes / 4096)
    references = pages * (sys.getsizeof([None]*4096) + struct.calcsize("P"))
    cache = 32 * 1024**2
    model_total = math.ceil(1.25 * (graph_estimate + references + cache)) + 1024**3
    host_forecast = math.ceil(model_total / 1024**3) * 1024**3
    # Upper bound on the compact row encoding: the first Fraction numerator is
    # at most bits+1 bits; all j/denominators are bounded by the proposed row count.
    shared = small["texture"]
    max_hex_digits = math.ceil((PROPOSED_ROWS + 1).bit_length() / 4)
    denominator_digits = math.ceil((2*PROPOSED_ROWS + 5).bit_length() / 4)
    bound_row = (PROPOSED_ROWS-1,
                 {"rational_hex": ("f"*(PROPOSED_RATIONAL_BITS//4+1),
                                    "f"*denominator_digits)},
                 {"rational_hex": ("-"+"f"*max_hex_digits, "f"*denominator_digits)},
                 {"shared": shared, "state": (True, None, "é")})
    import io
    out = io.StringIO()
    reference_dump(bound_row, out)
    max_row_bytes = len(out.getvalue().encode("utf-8"))
    output_bound = 2*PROPOSED_ROWS*(max_row_bytes+1) + 4096
    scratch_limit = 20*1024**3
    other_evidence = 64*1024**2
    return {
        "status": "FORECAST_ONLY_LARGE_BENCHMARK_NOT_AUTHORIZED_OR_RUN",
        "proposed_rows": PROPOSED_ROWS,
        "proposed_rational_prefix_bit": PROPOSED_RATIONAL_BITS,
        "maximum_constructed_numerator_bits": PROPOSED_RATIONAL_BITS+1,
        "predicted_unique_objects": nodes,
        "predicted_container_edges": edges,
        "predicted_fraction_occurrences": 8*PROPOSED_ROWS+2,
        "row_upper_encoded_bytes": max_row_bytes,
        "each_output_upper_bytes": output_bound,
        "index_database_limit_bytes": 16*1024**3,
        "index_and_journal_stop_limit_bytes": scratch_limit,
        "reference_pages": pages,
        "reference_shallow_estimate_bytes": references,
        "page_cache_target_bytes": cache,
        "small_sample_rows": [512, 1024],
        "small_sample_retained_shallow_bytes": [first_size, second_size],
        "extrapolated_graph_shallow_bytes": graph_estimate,
        "host_working_set_forecast_bytes": host_forecast,
        "minimum_fresh_available_RAM_bytes": host_forecast + 4*1024**3,
        "scratch_and_two_output_forecast_bytes": 2*output_bound + scratch_limit + other_evidence,
        "minimum_fresh_free_disk_bytes": 2*output_bound + scratch_limit + other_evidence + 1024**3,
        "proposed_one_job_timeout_seconds": 7200,
        "runtime_and_peak_commit_bytes": "not measured",
        "assumptions": [
            "Memory extrapolates small Python object sizes and adds25percent plus1GiB; not a hard bound.",
            "SQLite cache target excludes Python refs, SQLite metadata, OS file cache and journals.",
            "Two full emitted files permit streaming exact-byte comparison; neither is retained scientific output.",
            "Scratch checked each4096 inserts; filesystem write/commit failure refuses before destination IO.",
            "Fresh RAM, commit headroom, disk, quiet-process, source and retained-ledger admissions remain required.",
            "6144-bit numerators deliberately stress retained large rationals; actual history bit distribution unknown.",
            "Aliases appear through both coordinator and pixel paths; all histories in this synthetic fixture are emitted.",
            "Not a reconstruction of the absent failed coordinator history, and not whole-history or science acceptance.",
        ],
    }
