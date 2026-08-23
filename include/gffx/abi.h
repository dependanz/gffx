#ifndef GFFX_ABI_H
#define GFFX_ABI_H

#include <stdint.h>

#if UINTPTR_MAX != UINT64_MAX
#error "GFFX ABI v1 requires a 64-bit target"
#endif

#if defined(_WIN32)
#define GFFX_CALL __cdecl
#if defined(GFFX_BUILDING_LIBRARY)
#define GFFX_API __declspec(dllexport)
#else
#define GFFX_API __declspec(dllimport)
#endif
#else
#define GFFX_CALL
#define GFFX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define GFFX_EXTERN_C_BEGIN extern "C" {
#define GFFX_EXTERN_C_END }
#else
#define GFFX_EXTERN_C_BEGIN
#define GFFX_EXTERN_C_END
#endif

#define GFFX_ABI_VERSION_ENCODE(major, minor) \
    ((((uint32_t)(major)) << 16U) | ((uint32_t)(minor) & UINT32_C(0xffff)))
#define GFFX_ABI_VERSION_MAJOR(version) (((uint32_t)(version) >> 16U) & UINT32_C(0xffff))
#define GFFX_ABI_VERSION_MINOR(version) ((uint32_t)(version) & UINT32_C(0xffff))

#define GFFX_ABI_VERSION_MAJOR_CURRENT UINT32_C(1)
#define GFFX_ABI_VERSION_MINOR_CURRENT UINT32_C(0)
#define GFFX_ABI_VERSION \
    GFFX_ABI_VERSION_ENCODE(GFFX_ABI_VERSION_MAJOR_CURRENT, GFFX_ABI_VERSION_MINOR_CURRENT)

typedef uint32_t gffx_device_type;

#define GFFX_DEVICE_CPU UINT32_C(1)
#define GFFX_DEVICE_CUDA UINT32_C(2)

GFFX_EXTERN_C_BEGIN

GFFX_API uint32_t GFFX_CALL gffx_get_abi_version(void);

GFFX_EXTERN_C_END

#endif /* GFFX_ABI_H */
