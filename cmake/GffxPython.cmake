include_guard(GLOBAL)

# CPython Limited-API discovery and the private loader target.
#
# The extension is built against the CPython 3.10 stable ABI, so a single compiled artifact serves
# CPython 3.10 and later and the wheel carries an abi3 tag. Development.SABIModule provides the
# stable-ABI import library rather than the version-specific one.

# A build frontend (scikit-build-core) supplies an explicit interpreter plus the stable-ABI import
# library, but runs inside its own isolated virtual environment. FindPython defaults to preferring
# that active environment, which does not ship the development artifacts, so the standard search order must
# win. Only override when an interpreter was actually handed to us.
if(DEFINED Python_EXECUTABLE AND NOT DEFINED Python_FIND_VIRTUALENV)
    set(Python_FIND_VIRTUALENV STANDARD)
endif()

# scikit-build-core publishes the component name when a limited-ABI build was requested.
if(DEFINED SKBUILD_SABI_COMPONENT AND NOT SKBUILD_SABI_COMPONENT STREQUAL "")
    set(GFFX_SABI_COMPONENT "${SKBUILD_SABI_COMPONENT}")
else()
    set(GFFX_SABI_COMPONENT "Development.SABIModule")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module ${GFFX_SABI_COMPONENT})

python_add_library(gffx_python_core MODULE USE_SABI 3.10 WITH_SOABI
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/python/module.c
)

set_target_properties(gffx_python_core PROPERTIES
    OUTPUT_NAME "_core"
    C_VISIBILITY_PRESET hidden
)

target_link_libraries(gffx_python_core PRIVATE gffx_core)

# The extension and the core ship side by side inside the installed package. Windows finds the
# neighbouring DLL for free because CPython loads extensions with an altered search path, but ELF
# and Mach-O need an explicit runtime search path relative to the loaded module. Without this,
# auditwheel cannot resolve libgffx_core.so and the Linux wheel fails to repair.
if(APPLE)
    set_target_properties(gffx_python_core PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH ON
    )
elseif(UNIX)
    set_target_properties(gffx_python_core PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH ON
    )
endif()

# The loader sits inside the installed package next to gffx_core, which the platform loader finds
# because CPython loads extension modules with an altered search path rooted at the module.
install(TARGETS gffx_python_core
    RUNTIME DESTINATION gffx
    LIBRARY DESTINATION gffx
)
