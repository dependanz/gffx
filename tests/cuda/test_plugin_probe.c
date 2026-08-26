#include <gffx/capabilities.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

static int set_plugin_path(const char *path) {
#if defined(_WIN32)
    return _putenv_s("GFFX_CUDA_PLUGIN_PATH", path);
#else
    return setenv("GFFX_CUDA_PLUGIN_PATH", path, 1);
#endif
}

static const gffx_capability_record *find_key(
    const gffx_capability_report *report,
    uint32_t key
) {
    uint64_t index;
    for (index = 0u; index < report->record_count; ++index) {
        if (report->records[index].key == key) return &report->records[index];
    }
    return NULL;
}

static const char *record_string(
    const gffx_capability_report *report,
    const gffx_capability_record *record
) {
    if (record == NULL || record->value_type != GFFX_CAPABILITY_VALUE_STRING ||
        record->string_offset >= report->string_size_bytes) return NULL;
    return report->strings + record->string_offset;
}

int main(int argc, char **argv) {
    gffx_capability_record records[256] = {{0}};
    char strings[16384] = {0};
    gffx_capability_report report = {0};
    const gffx_capability_record *status_record;
    const gffx_capability_record *path_record;
    const gffx_capability_record *compatible_record;
    const char *status_text;
    gffx_status status;

    CHECK(argc == 4);
    if (strcmp(argv[1], "DEFAULT") == 0) {
        CHECK(set_plugin_path("") == 0);
    } else {
        CHECK(set_plugin_path(argv[1]) == 0);
    }

    report.struct_size = (uint32_t)sizeof(report);
    report.abi_version = GFFX_ABI_VERSION;
    status = gffx_capabilities_probe(GFFX_CAPABILITY_PROBE_FULL, &report, NULL);
    CHECK(status == GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    CHECK(report.required_record_count > UINT64_C(0));
    CHECK(report.required_record_count <= UINT64_C(256));
    CHECK(report.required_string_bytes <= (uint64_t)sizeof(strings));

    memset(&report, 0, sizeof(report));
    report.struct_size = (uint32_t)sizeof(report);
    report.abi_version = GFFX_ABI_VERSION;
    report.records = records;
    report.record_capacity = UINT64_C(256);
    report.strings = strings;
    report.string_capacity_bytes = (uint64_t)sizeof(strings);
    status = gffx_capabilities_probe(GFFX_CAPABILITY_PROBE_FULL, &report, NULL);
    CHECK(status == GFFX_STATUS_OK);

    status_record = find_key(&report, GFFX_CAPABILITY_KEY_CUDA_PROVIDER_STATUS);
    path_record = find_key(&report, GFFX_CAPABILITY_KEY_CUDA_PLUGIN_PATH);
    compatible_record = find_key(&report, GFFX_CAPABILITY_KEY_CUDA_PLUGIN_COMPATIBLE);
    status_text = record_string(&report, status_record);
    CHECK(status_text != NULL);
    CHECK(strstr(status_text, argv[2]) != NULL);
    CHECK(path_record != NULL);
    CHECK((path_record->flags & GFFX_CAPABILITY_RECORD_SENSITIVE) != 0u);
    if (strcmp(argv[1], "DEFAULT") == 0) {
        const char *discovered_path = record_string(&report, path_record);
        CHECK(strstr(discovered_path, "gffx_cuda12") != NULL);
        CHECK(strstr(discovered_path, ".dll") != NULL || strstr(discovered_path, ".so") != NULL);
    } else {
        CHECK(strcmp(record_string(&report, path_record), argv[1]) == 0);
    }

    if (strcmp(argv[3], "compatible") == 0) {
        const gffx_capability_record *driver_status =
            find_key(&report, GFFX_CAPABILITY_KEY_CUDA_DRIVER_STATUS);
        CHECK(compatible_record != NULL);
        CHECK(compatible_record->value_type == GFFX_CAPABILITY_VALUE_BOOL);
        CHECK(compatible_record->value_u64 == UINT64_C(1));
        CHECK(driver_status != NULL);
        CHECK(strstr(record_string(&report, driver_status), "synthetic") != NULL);
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_OPTIONAL_PROVIDER_ABSENT) == 0u);
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE) == 0u);
    } else if (strcmp(argv[3], "real") == 0) {
        const gffx_capability_record *driver_status =
            find_key(&report, GFFX_CAPABILITY_KEY_CUDA_DRIVER_STATUS);
        const gffx_capability_record *device_count =
            find_key(&report, GFFX_CAPABILITY_KEY_CUDA_DEVICE_COUNT);
        const gffx_capability_record *device_name =
            find_key(&report, GFFX_CAPABILITY_KEY_CUDA_DEVICE_NAME);
        const gffx_capability_record *compute_major =
            find_key(&report, GFFX_CAPABILITY_KEY_CUDA_COMPUTE_CAPABILITY_MAJOR);
        CHECK(compatible_record != NULL);
        CHECK(compatible_record->value_u64 == UINT64_C(1));
        CHECK(driver_status != NULL);
        CHECK(strstr(record_string(&report, driver_status), "available") != NULL);
        CHECK(device_count != NULL);
        CHECK(device_count->value_u64 >= UINT64_C(1));
        CHECK(device_name != NULL);
        CHECK(record_string(&report, device_name) != NULL);
        CHECK(compute_major != NULL);
        CHECK(compute_major->value_u64 >= UINT64_C(7));
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_OPTIONAL_PROVIDER_ABSENT) == 0u);
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE) == 0u);
    } else if (strcmp(argv[3], "partial") == 0) {
        CHECK(compatible_record != NULL);
        CHECK(compatible_record->value_u64 == UINT64_C(0));
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE) != 0u);
    } else {
        CHECK(strcmp(argv[3], "absent") == 0);
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_OPTIONAL_PROVIDER_ABSENT) != 0u);
        CHECK((report.result_flags & GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE) == 0u);
    }
    return 0;
}
