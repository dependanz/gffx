#include <gffx/capabilities.h>

#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

static gffx_capability_report make_report(
    gffx_capability_record *records,
    uint64_t record_capacity,
    char *strings,
    uint64_t string_capacity
) {
    gffx_capability_report report = {0};
    report.struct_size = (uint32_t)sizeof(report);
    report.abi_version = GFFX_ABI_VERSION;
    report.records = records;
    report.record_capacity = record_capacity;
    report.strings = strings;
    report.string_capacity_bytes = string_capacity;
    return report;
}

static int valid_string_record(
    const gffx_capability_record *record,
    const gffx_capability_report *report
) {
    if (record->value_type != GFFX_CAPABILITY_VALUE_STRING) return 1;
    if (record->string_offset > report->string_size_bytes) return 0;
    if (record->string_size > report->string_size_bytes - record->string_offset) return 0;
    if (record->string_size == 0u) return 0;
    return report->strings[record->string_offset + record->string_size - 1u] == '\0';
}

static int find_key(
    const gffx_capability_report *report,
    uint32_t key,
    const gffx_capability_record **found
) {
    uint64_t index;
    for (index = 0u; index < report->record_count; ++index) {
        if (report->records[index].key == key) {
            *found = &report->records[index];
            return 1;
        }
    }
    return 0;
}

static int run_query(int full_probe) {
    gffx_capability_record records[32] = {{0}};
    char strings[1024] = {0};
    gffx_capability_report report = make_report(NULL, 0u, NULL, 0u);
    const gffx_capability_record *record = NULL;
    gffx_status status;
    uint64_t required_records;
    uint64_t required_strings;
    uint64_t index;

    status = full_probe
        ? gffx_capabilities_probe(GFFX_CAPABILITY_PROBE_FULL, &report, NULL)
        : gffx_capabilities_query(&report, NULL);
    CHECK(status == GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    CHECK(report.record_count == 0u);
    CHECK(report.string_size_bytes == 0u);
    CHECK(report.required_record_count > 0u);
    CHECK(report.required_string_bytes > 0u);
    required_records = report.required_record_count;
    required_strings = report.required_string_bytes;
    CHECK(required_records <= 32u);
    CHECK(required_strings <= sizeof(strings));

    report = make_report(records, 32u, strings, sizeof(strings));
    status = full_probe
        ? gffx_capabilities_probe(GFFX_CAPABILITY_PROBE_FULL, &report, NULL)
        : gffx_capabilities_query(&report, NULL);
    CHECK(status == GFFX_STATUS_OK);
    CHECK(report.record_count == required_records);
    CHECK(report.required_record_count == required_records);
    CHECK(report.string_size_bytes == required_strings);
    CHECK(report.required_string_bytes == required_strings);

    for (index = 0u; index < report.record_count; ++index) {
        CHECK(report.records[index].struct_size == sizeof(gffx_capability_record));
        CHECK(report.records[index].abi_version == GFFX_ABI_VERSION);
        CHECK(valid_string_record(&report.records[index], &report));
    }

    CHECK(find_key(&report, GFFX_CAPABILITY_KEY_ABI_VERSION, &record));
    CHECK(record->value_type == GFFX_CAPABILITY_VALUE_U64);
    CHECK(record->value_u64 == GFFX_ABI_VERSION);
    CHECK(find_key(&report, GFFX_CAPABILITY_KEY_PACKAGE_VERSION, &record));
    CHECK(record->value_type == GFFX_CAPABILITY_VALUE_STRING);
    CHECK(strcmp(report.strings + record->string_offset, "0.2.0.dev0") == 0);
    CHECK(find_key(&report, GFFX_CAPABILITY_KEY_POINTER_BITS, &record));
    CHECK(record->value_u64 == 64u);

    if (full_probe) {
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_RUNTIME_PROBED) != 0u);
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_OPTIONAL_PROVIDER_ABSENT) != 0u);
        CHECK(find_key(&report, GFFX_CAPABILITY_KEY_CUDA_PROVIDER_STATUS, &record));
        CHECK(strcmp(report.strings + record->string_offset, "not built") == 0);
    } else {
        CHECK(report.result_flags == GFFX_CAPABILITY_RESULT_STATIC);
        CHECK(!find_key(&report, GFFX_CAPABILITY_KEY_CUDA_PROVIDER_STATUS, &record));
    }
    return 0;
}

int main(void) {
    int result = run_query(0);
    if (result != 0) return result;
    result = run_query(1);
    if (result != 0) return result;
    return 0;
}
