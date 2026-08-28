#if !defined(_WIN32)
#define _GNU_SOURCE 1
#endif

#include "cuda_loader.h"
#include "plugin_api.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HMODULE gffx_library_handle;
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
typedef void *gffx_library_handle;
#endif

#define GFFX_CUDA_PATH_CAPACITY 4096u
#define GFFX_CUDA_STATUS_CAPACITY 1024u

typedef enum gffx_cuda_loader_kind {
    GFFX_CUDA_LOADER_ABSENT = 0,
    GFFX_CUDA_LOADER_FAILURE = 1,
    GFFX_CUDA_LOADER_READY = 2
} gffx_cuda_loader_kind;

typedef struct gffx_cuda_loader_state {
    gffx_cuda_loader_kind kind;
    char path[GFFX_CUDA_PATH_CAPACITY];
    char status[GFFX_CUDA_STATUS_CAPACITY];
    gffx_library_handle library;
    gffx_cuda_plugin_api api;
    uint64_t provider_records;
    uint64_t provider_strings;
    uint32_t provider_result_flags;
} gffx_cuda_loader_state;

typedef struct gffx_loader_spec {
    uint32_t key;
    uint32_t value_type;
    uint32_t flags;
    uint64_t value_u64;
    const char *string_value;
} gffx_loader_spec;

static void copy_text(char *destination, size_t capacity, const char *source) {
    if (capacity == 0u) return;
    if (source == NULL) source = "";
#if defined(_MSC_VER)
    (void)strncpy_s(destination, capacity, source, _TRUNCATE);
#else
    (void)snprintf(destination, capacity, "%s", source);
#endif
}

static void format_status(
    char *destination,
    size_t capacity,
    const char *prefix,
    const char *detail
) {
#if defined(_MSC_VER)
    (void)_snprintf_s(destination, capacity, _TRUNCATE, "%s%s%s", prefix,
        (detail != NULL && detail[0] != '\0') ? ": " : "", detail != NULL ? detail : "");
#else
    (void)snprintf(destination, capacity, "%s%s%s", prefix,
        (detail != NULL && detail[0] != '\0') ? ": " : "", detail != NULL ? detail : "");
#endif
}

static int is_absolute_path(const char *path) {
#if defined(_WIN32)
    if (path == NULL) return 0;
    return ((path[0] != '\0' && path[1] == ':' &&
             (path[2] == '\\' || path[2] == '/')) ||
            (path[0] == '\\' && path[1] == '\\'));
#else
    return path != NULL && path[0] == '/';
#endif
}

static int module_path(char *path, size_t capacity) {
#if defined(_WIN32)
    HMODULE module = NULL;
    DWORD length;
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)(uintptr_t)&gffx_cuda_loader_probe,
            &module)) return 0;
    length = GetModuleFileNameA(module, path, (DWORD)capacity);
    return length != 0u && (size_t)length < capacity;
#else
    Dl_info information;
    void *address = NULL;
    gffx_status (*function_pointer)(uint32_t, gffx_capability_report *,
                                    gffx_diagnostic_buffer *) = gffx_cuda_loader_probe;
    memcpy(&address, &function_pointer, sizeof(address));
    if (dladdr(address, &information) == 0 || information.dli_fname == NULL) return 0;
    if (realpath(information.dli_fname, path) == NULL) return 0;
    return strlen(path) < capacity;
#endif
}

static int default_path(char *path, size_t capacity) {
    char loaded_module[GFFX_CUDA_PATH_CAPACITY];
    char *separator;
    const char *filename;
    size_t directory_size;
    if (!module_path(loaded_module, sizeof(loaded_module))) return 0;
#if defined(_WIN32)
    separator = strrchr(loaded_module, '\\');
    filename = "gffx_cuda12.dll";
#elif defined(__APPLE__)
    separator = strrchr(loaded_module, '/');
    filename = "libgffx_cuda12.dylib";
#else
    separator = strrchr(loaded_module, '/');
    filename = "libgffx_cuda12.so";
#endif
    if (separator == NULL) return 0;
    directory_size = (size_t)(separator - loaded_module) + 1u;
    if (directory_size + strlen(filename) + 1u > capacity) return 0;
    memcpy(path, loaded_module, directory_size);
    memcpy(path + directory_size, filename, strlen(filename) + 1u);
    return 1;
}

static int path_exists(const char *path) {
#if defined(_WIN32)
    DWORD attributes = GetFileAttributesA(path);
    return attributes != INVALID_FILE_ATTRIBUTES &&
           (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0u;
#else
    return access(path, F_OK) == 0;
#endif
}

static int has_shared_library_magic(const char *path) {
    unsigned char bytes[4] = {0u, 0u, 0u, 0u};
#if defined(_WIN32)
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD bytes_read = 0u;
    if (file == INVALID_HANDLE_VALUE) return 0;
    if (!ReadFile(file, bytes, (DWORD)sizeof(bytes), &bytes_read, NULL)) bytes_read = 0u;
    (void)CloseHandle(file);
    return bytes_read >= 2u && bytes[0] == 'M' && bytes[1] == 'Z';
#else
    int file = open(path, O_RDONLY);
    ssize_t bytes_read;
    uint32_t magic;
    if (file < 0) return 0;
    bytes_read = read(file, bytes, sizeof(bytes));
    (void)close(file);
    if (bytes_read != (ssize_t)sizeof(bytes)) return 0;
#if defined(__APPLE__)
    memcpy(&magic, bytes, sizeof(magic));
    return magic == UINT32_C(0xfeedface) || magic == UINT32_C(0xfeedfacf) ||
           magic == UINT32_C(0xcefaedfe) || magic == UINT32_C(0xcffaedfe) ||
           magic == UINT32_C(0xcafebabe) || magic == UINT32_C(0xbebafeca);
#else
    (void)magic;
    return bytes[0] == UINT8_C(0x7f) && bytes[1] == 'E' &&
           bytes[2] == 'L' && bytes[3] == 'F';
#endif
#endif
}

static void last_loader_error(char *detail, size_t capacity) {
#if defined(_WIN32)
    DWORD error_code = GetLastError();
    DWORD result = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, error_code, 0, detail, (DWORD)capacity, NULL);
    if (result == 0u) {
        (void)_snprintf_s(detail, capacity, _TRUNCATE, "Win32 error %lu", error_code);
    } else {
        while (result > 0u && (detail[result - 1u] == '\r' || detail[result - 1u] == '\n')) {
            detail[--result] = '\0';
        }
    }
#else
    const char *error_text = dlerror();
    copy_text(detail, capacity, error_text != NULL ? error_text : "dynamic loader error");
#endif
}

static gffx_library_handle open_library(const char *path) {
#if defined(_WIN32)
    return LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void close_library(gffx_library_handle library) {
    if (library == NULL) return;
#if defined(_WIN32)
    (void)FreeLibrary(library);
#else
    (void)dlclose(library);
#endif
}

static gffx_cuda_plugin_handshake_fn handshake_symbol(gffx_library_handle library) {
    gffx_cuda_plugin_handshake_fn function_pointer = NULL;
#if defined(_WIN32)
    FARPROC symbol = GetProcAddress(library, GFFX_CUDA_PLUGIN_HANDSHAKE_SYMBOL);
    memcpy(&function_pointer, &symbol, sizeof(function_pointer));
#else
    void *symbol;
    (void)dlerror();
    symbol = dlsym(library, GFFX_CUDA_PLUGIN_HANDSHAKE_SYMBOL);
    memcpy(&function_pointer, &symbol, sizeof(function_pointer));
#endif
    return function_pointer;
}

static int api_is_compatible(const gffx_cuda_plugin_api *api, char *detail, size_t capacity) {
    uint64_t index;
    const size_t required_size = offsetof(gffx_cuda_plugin_api, capabilities_probe) +
                                 sizeof(api->capabilities_probe);
    if (api->struct_size < required_size) {
        copy_text(detail, capacity, "plugin API structure is too small");
        return 0;
    }
    if (GFFX_ABI_VERSION_MAJOR(api->plugin_abi_version) !=
            GFFX_CUDA_PLUGIN_ABI_VERSION_MAJOR ||
        GFFX_ABI_VERSION_MINOR(api->plugin_abi_version) >
            GFFX_CUDA_PLUGIN_ABI_VERSION_MINOR) {
        copy_text(detail, capacity, "unsupported CUDA plugin ABI version");
        return 0;
    }
    if (GFFX_ABI_VERSION < api->core_abi_min || GFFX_ABI_VERSION > api->core_abi_max) {
        copy_text(detail, capacity, "plugin does not support core ABI 1.0");
        return 0;
    }
    if ((api->flags & GFFX_CUDA_PLUGIN_FLAG_CAPABILITY_PROVIDER) == 0u ||
        api->capabilities_probe == NULL) {
        copy_text(detail, capacity, "plugin has no capability provider");
        return 0;
    }
    if (api->build_identity == NULL || api->build_identity[0] == '\0') {
        copy_text(detail, capacity, "plugin build identity is empty");
        return 0;
    }
    /* The operation table is optional; a plugin may provide capabilities only. What is not
     * optional is consistency: advertising the flag without a usable table, or publishing a table
     * too small to contain the fields this host reads, is a build mismatch rather than a
     * capability the host can work around. */
    if ((api->flags & GFFX_CUDA_PLUGIN_FLAG_OPERATION_PROVIDER) != 0u) {
        const size_t required_operations_size =
            offsetof(gffx_cuda_operations, render_interpolate_backward) +
            sizeof(((const gffx_cuda_operations *)0)->render_interpolate_backward);
        if (api->struct_size < offsetof(gffx_cuda_plugin_api, operations) +
                                   sizeof(api->operations)) {
            copy_text(detail, capacity,
                      "plugin advertises operations but its API struct predates the field");
            return 0;
        }
        if (api->operations == NULL) {
            copy_text(detail, capacity, "plugin advertises operations but published no table");
            return 0;
        }
        if (api->operations->struct_size < required_operations_size) {
            copy_text(detail, capacity, "plugin operation table is smaller than this host reads");
            return 0;
        }
    }
    if (api->struct_size >= sizeof(*api)) {
        /* Five, not six: the operations pointer took one slot from the reserved tail when
         * dispatch joined v1, so scanning six would read past the end of the struct. */
        for (index = 0u; index < 5u; ++index) {
            if (api->reserved[index] != UINT64_C(0)) {
                copy_text(detail, capacity, "plugin API reserved fields are nonzero");
                return 0;
            }
        }
    }
    return 1;
}

static void prepare_state(gffx_cuda_loader_state *state, uint32_t probe_flags) {
    const char *override_path = getenv("GFFX_CUDA_PLUGIN_PATH");
    char detail[GFFX_CUDA_STATUS_CAPACITY] = {0};
    char handshake_text[GFFX_CUDA_STATUS_CAPACITY] = {0};
    gffx_diagnostic_buffer diagnostic = GFFX_DIAGNOSTIC_INIT;
    gffx_cuda_plugin_handshake_fn handshake;
    gffx_capability_report provider_report = {0};
    gffx_status status;

    memset(state, 0, sizeof(*state));
    diagnostic.data = handshake_text;
    diagnostic.capacity_bytes = (uint64_t)sizeof(handshake_text);
    if (override_path != NULL && override_path[0] != '\0') {
        if (strlen(override_path) + 1u > sizeof(state->path)) {
            state->kind = GFFX_CUDA_LOADER_FAILURE;
            copy_text(state->path, sizeof(state->path), "<path too long>");
            copy_text(state->status, sizeof(state->status),
                      "invalid explicit path: exceeds loader capacity");
            return;
        }
        copy_text(state->path, sizeof(state->path), override_path);
        if (!is_absolute_path(state->path)) {
            state->kind = GFFX_CUDA_LOADER_FAILURE;
            copy_text(state->status, sizeof(state->status),
                      "invalid explicit path: an absolute path is required");
            return;
        }
    } else if (!default_path(state->path, sizeof(state->path))) {
        state->kind = GFFX_CUDA_LOADER_FAILURE;
        copy_text(state->path, sizeof(state->path), "<unresolved>");
        copy_text(state->status, sizeof(state->status),
                  "discovery failed: could not resolve the core library directory");
        return;
    }
    if (!path_exists(state->path)) {
        state->kind = GFFX_CUDA_LOADER_ABSENT;
        copy_text(state->status, sizeof(state->status), "not found");
        return;
    }
    if (!has_shared_library_magic(state->path)) {
        state->kind = GFFX_CUDA_LOADER_FAILURE;
        copy_text(state->status, sizeof(state->status),
                  "load failed: invalid shared-library header");
        return;
    }
    state->library = open_library(state->path);
    if (state->library == NULL) {
        state->kind = GFFX_CUDA_LOADER_FAILURE;
        last_loader_error(detail, sizeof(detail));
        format_status(state->status, sizeof(state->status), "load failed", detail);
        return;
    }
    handshake = handshake_symbol(state->library);
    if (handshake == NULL) {
        state->kind = GFFX_CUDA_LOADER_FAILURE;
        copy_text(state->status, sizeof(state->status),
                  "incompatible: missing handshake symbol " GFFX_CUDA_PLUGIN_HANDSHAKE_SYMBOL);
        return;
    }
    memset(&state->api, 0, sizeof(state->api));
    state->api.struct_size = (uint32_t)sizeof(state->api);
    status = handshake(GFFX_CUDA_PLUGIN_ABI_VERSION, GFFX_ABI_VERSION,
                       &state->api, &diagnostic);
    if (status != GFFX_STATUS_OK) {
        state->kind = GFFX_CUDA_LOADER_FAILURE;
        format_status(state->status, sizeof(state->status),
                      "incompatible: handshake rejected", handshake_text);
        return;
    }
    if (!api_is_compatible(&state->api, detail, sizeof(detail))) {
        state->kind = GFFX_CUDA_LOADER_FAILURE;
        format_status(state->status, sizeof(state->status), "incompatible", detail);
        return;
    }
    provider_report.struct_size = (uint32_t)sizeof(provider_report);
    provider_report.abi_version = GFFX_ABI_VERSION;
    status = state->api.capabilities_probe(probe_flags, &provider_report, &diagnostic);
    if (status != GFFX_STATUS_INSUFFICIENT_WORKSPACE && status != GFFX_STATUS_OK) {
        state->kind = GFFX_CUDA_LOADER_FAILURE;
        format_status(state->status, sizeof(state->status),
                      "loaded; capability probe failed", handshake_text);
        return;
    }
    state->kind = GFFX_CUDA_LOADER_READY;
    copy_text(state->status, sizeof(state->status), "loaded");
    state->provider_records = provider_report.required_record_count;
    state->provider_strings = provider_report.required_string_bytes;
    state->provider_result_flags = provider_report.result_flags;
}

static uint64_t spec_string_size(const gffx_loader_spec *spec) {
    return spec->value_type == GFFX_CAPABILITY_VALUE_STRING
        ? (uint64_t)strlen(spec->string_value) + UINT64_C(1) : UINT64_C(0);
}

static void emit_spec(const gffx_loader_spec *spec, gffx_capability_record *record,
                      char *strings, uint64_t *string_cursor) {
    uint64_t string_size = spec_string_size(spec);
    memset(record, 0, sizeof(*record));
    record->struct_size = (uint32_t)sizeof(*record);
    record->abi_version = GFFX_ABI_VERSION;
    record->category = GFFX_CAPABILITY_CATEGORY_BACKEND;
    record->key = spec->key;
    record->value_type = spec->value_type;
    record->flags = spec->flags;
    record->value_u64 = spec->value_u64;
    if (string_size != UINT64_C(0)) {
        record->string_offset = *string_cursor;
        record->string_size = string_size;
        memcpy(strings + *string_cursor, spec->string_value, (size_t)string_size);
        *string_cursor += string_size;
    }
}

gffx_status gffx_cuda_loader_probe(uint32_t probe_flags, gffx_capability_report *report,
                                   gffx_diagnostic_buffer *diagnostic) {
    gffx_cuda_loader_state state;
    gffx_loader_spec specs[5];
    uint64_t metadata_count = UINT64_C(3);
    uint64_t metadata_strings = UINT64_C(0);
    uint64_t required_records;
    uint64_t required_strings;
    uint64_t string_cursor = UINT64_C(0);
    uint64_t index;
    (void)diagnostic;
    if (report == NULL || report->struct_size < sizeof(*report) ||
        report->abi_version != GFFX_ABI_VERSION) return GFFX_STATUS_INVALID_ARGUMENT;

    prepare_state(&state, probe_flags);
    specs[0] = (gffx_loader_spec){GFFX_CAPABILITY_KEY_CUDA_PROVIDER_STATUS,
        GFFX_CAPABILITY_VALUE_STRING, 0u, UINT64_C(0), state.status};
    specs[1] = (gffx_loader_spec){GFFX_CAPABILITY_KEY_CUDA_PLUGIN_PATH,
        GFFX_CAPABILITY_VALUE_STRING, GFFX_CAPABILITY_RECORD_SENSITIVE,
        UINT64_C(0), state.path};
    specs[2] = (gffx_loader_spec){GFFX_CAPABILITY_KEY_CUDA_PLUGIN_COMPATIBLE,
        GFFX_CAPABILITY_VALUE_BOOL, 0u,
        state.kind == GFFX_CUDA_LOADER_READY ? UINT64_C(1) : UINT64_C(0), NULL};
    if (state.api.plugin_abi_version != UINT32_C(0)) {
        specs[metadata_count++] = (gffx_loader_spec){
            GFFX_CAPABILITY_KEY_CUDA_PLUGIN_ABI_VERSION, GFFX_CAPABILITY_VALUE_U64,
            0u, state.api.plugin_abi_version, NULL};
    }
    if (state.kind == GFFX_CUDA_LOADER_READY) {
        specs[metadata_count++] = (gffx_loader_spec){
            GFFX_CAPABILITY_KEY_CUDA_PLUGIN_BUILD_ID, GFFX_CAPABILITY_VALUE_STRING,
            0u, UINT64_C(0), state.api.build_identity};
    }
    for (index = 0u; index < metadata_count; ++index) {
        metadata_strings += spec_string_size(&specs[index]);
    }
    required_records = metadata_count +
        (state.kind == GFFX_CUDA_LOADER_READY ? state.provider_records : UINT64_C(0));
    required_strings = metadata_strings +
        (state.kind == GFFX_CUDA_LOADER_READY ? state.provider_strings : UINT64_C(0));
    report->record_count = UINT64_C(0);
    report->string_size_bytes = UINT64_C(0);
    report->required_record_count = required_records;
    report->required_string_bytes = required_strings;
    report->query_flags = probe_flags;
    report->result_flags = GFFX_CAPABILITY_RESULT_RUNTIME_PROBED;
    if (state.kind == GFFX_CUDA_LOADER_ABSENT) {
        report->result_flags |= GFFX_CAPABILITY_RESULT_OPTIONAL_PROVIDER_ABSENT;
    } else if (state.kind == GFFX_CUDA_LOADER_FAILURE) {
        report->result_flags |= GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE;
    } else {
        report->result_flags |= state.provider_result_flags;
    }
    if (report->records == NULL || report->record_capacity < required_records ||
        report->strings == NULL || report->string_capacity_bytes < required_strings) {
        close_library(state.library);
        return GFFX_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (state.kind == GFFX_CUDA_LOADER_READY && state.provider_records != UINT64_C(0)) {
        gffx_capability_report provider = {0};
        gffx_status provider_status;
        provider.struct_size = (uint32_t)sizeof(provider);
        provider.abi_version = GFFX_ABI_VERSION;
        provider.records = report->records + metadata_count;
        provider.record_capacity = report->record_capacity - metadata_count;
        provider.strings = report->strings + metadata_strings;
        provider.string_capacity_bytes = report->string_capacity_bytes - metadata_strings;
        provider_status = state.api.capabilities_probe(probe_flags, &provider, diagnostic);
        if (provider_status == GFFX_STATUS_OK) {
            for (index = 0u; index < provider.record_count; ++index) {
                if (provider.records[index].value_type == GFFX_CAPABILITY_VALUE_STRING) {
                    provider.records[index].string_offset += metadata_strings;
                }
            }
            state.provider_records = provider.record_count;
            state.provider_strings = provider.string_size_bytes;
            report->result_flags |= provider.result_flags;
        } else {
            state.provider_records = UINT64_C(0);
            state.provider_strings = UINT64_C(0);
            report->result_flags |= GFFX_CAPABILITY_RESULT_PARTIAL_FAILURE;
            /* Keep the pre-sized status string length stable while reporting a partial result. */
            copy_text(state.status, sizeof(state.status), "failed");
        }
    }
    for (index = 0u; index < metadata_count; ++index) {
        if (index == 0u) specs[index].string_value = state.status;
        emit_spec(&specs[index], &report->records[index], report->strings, &string_cursor);
    }
    report->record_count = metadata_count + state.provider_records;
    report->string_size_bytes = metadata_strings + state.provider_strings;
    report->required_record_count = report->record_count;
    report->required_string_bytes = report->string_size_bytes;
    close_library(state.library);
    return GFFX_STATUS_OK;
}
