/*
 * Proves the Phase 1 Step 5 private calling path:
 *
 *   argv[1] = the private self-test build of the core. Its private entry point must resolve
 *             across a real shared-library boundary and report every check.
 *   argv[2] = the default shipped core. The same private entry point must be ABSENT from it,
 *             which is what keeps the self-test out of the public ABI v1 surface.
 */

#include "self_test.h"

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef gffx_status(GFFX_CALL *gffx_self_test_fn)(uint32_t *, gffx_diagnostic_buffer *);

#if defined(_WIN32)
typedef HMODULE gffx_library;
#define GFFX_LOAD(path) LoadLibraryA(path)
#define GFFX_RESOLVE(library, name) ((void *)GetProcAddress((library), (name)))
#define GFFX_CLOSE(library) FreeLibrary(library)
#else
typedef void *gffx_library;
#define GFFX_LOAD(path) dlopen((path), RTLD_NOW | RTLD_LOCAL)
#define GFFX_RESOLVE(library, name) dlsym((library), (name))
#define GFFX_CLOSE(library) dlclose(library)
#endif

static const char *const private_symbol = "gffx_private_self_test";

int main(int argc, char **argv) {
    gffx_library self_test_library;
    gffx_library default_library;
    gffx_self_test_fn self_test;
    void *resolved;
    uint32_t checks = UINT32_C(0);
    char message[256] = {0};
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_status status;

    if (argc != 3) return 1;

    self_test_library = GFFX_LOAD(argv[1]);
    if (self_test_library == NULL) return 2;

    resolved = GFFX_RESOLVE(self_test_library, private_symbol);
    if (resolved == NULL) {
        GFFX_CLOSE(self_test_library);
        return 3;
    }

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    diagnostic.data = message;
    diagnostic.capacity_bytes = (uint64_t)sizeof(message);

    self_test = (gffx_self_test_fn)resolved;
    status = self_test(&checks, &diagnostic);
    GFFX_CLOSE(self_test_library);

    if (status != GFFX_STATUS_OK) return 4;
    if (checks != GFFX_SELF_TEST_ALL) return 5;
    if (message[0] != '\0') return 6;

    /* The private entry point must not be reachable from the shipped default library. */
    default_library = GFFX_LOAD(argv[2]);
    if (default_library == NULL) return 7;
    resolved = GFFX_RESOLVE(default_library, private_symbol);
    GFFX_CLOSE(default_library);
    if (resolved != NULL) return 8;

    return 0;
}
