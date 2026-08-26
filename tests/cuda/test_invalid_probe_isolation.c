#include <gffx/capabilities.h>

#include <stdlib.h>

static int set_plugin_path(const char *path) {
#if defined(_WIN32)
    return _putenv_s("GFFX_CUDA_PLUGIN_PATH", path);
#else
    return setenv("GFFX_CUDA_PLUGIN_PATH", path, 1);
#endif
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    if (set_plugin_path(argv[1]) != 0) return 2;
    /* The synthetic plugin aborts if its handshake is called. Validation must happen first. */
    return gffx_capabilities_probe(GFFX_CAPABILITY_PROBE_FULL, NULL, NULL) ==
            GFFX_STATUS_INVALID_ARGUMENT
        ? 0 : 3;
}
