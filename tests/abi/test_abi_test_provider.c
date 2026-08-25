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

int main(void) {
    gffx_capability_record records[256] = {{0}};
    char strings[16384] = {0};
    gffx_capability_report report = make_report(NULL, 0u, NULL, 0u);
    uint64_t required_records;
    uint64_t required_strings;
    uint64_t index;
    int found_failure = 0;

    CHECK(gffx_capabilities_probe(GFFX_CAPABILITY_PROBE_FULL, &report, NULL) ==
          GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    required_records = report.required_record_count;
    required_strings = report.required_string_bytes;
    CHECK(required_records <= 256u);
    CHECK(required_strings <= sizeof(strings));

    report = make_report(records, 256u, strings, sizeof(strings));
    CHECK(gffx_capabilities_probe(GFFX_CAPABILITY_PROBE_FULL, &report, NULL) == GFFX_STATUS_OK);
    CHECK(report.record_count == required_records);
    CHECK(report.string_size_bytes == required_strings);
    CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_RUNTIME_PROBED) != 0u);
    CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_OPTIONAL_PROVIDER_ABSENT) != 0u);
    CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE) != 0u);

    for (index = 0u; index < report.record_count; ++index) {
        const gffx_capability_record *record = &report.records[index];
        if (record->key == GFFX_CAPABILITY_KEY_PROVIDER_STATUS) {
            CHECK(record->value_type == GFFX_CAPABILITY_VALUE_STRING);
            CHECK(strcmp(strings + record->string_offset, "test provider: synthetic failure") == 0);
            found_failure = 1;
        }
    }
    CHECK(found_failure);
    return 0;
}
