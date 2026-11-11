# Fail immediately when the Windows PTX resource generator did not produce its
# declared files. In particular, MSBuild can treat an empty custom COMMAND as
# successful while logging MSB8065 for every missing OUTPUT.
cmake_minimum_required(VERSION 3.25)

foreach(required IN ITEMS
    GFFX_PTX_RESOURCE_HEADER
    GFFX_PTX_RESOURCE_PAYLOAD
    GFFX_PTX_RESOURCE_SCRIPT
)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Missing PTX resource output argument: ${required}")
    endif()
    if(NOT EXISTS "${${required}}")
        message(FATAL_ERROR "PTX resource generator did not create ${required}: ${${required}}")
    endif()
    file(SIZE "${${required}}" generated_bytes)
    if(generated_bytes LESS 1)
        message(FATAL_ERROR "PTX resource generator wrote an empty output: ${${required}}")
    endif()
endforeach()

file(STRINGS "${GFFX_PTX_RESOURCE_HEADER}" resource_id_line
    REGEX "^#define GFFX_PTX_RESOURCE_ID [0-9]+$"
)
if(NOT resource_id_line)
    message(FATAL_ERROR "Generated PTX header lacks its resource identifier")
endif()

file(STRINGS "${GFFX_PTX_RESOURCE_SCRIPT}" resource_registration
    REGEX "^[0-9]+ RCDATA "
)
if(NOT resource_registration)
    message(FATAL_ERROR "Generated PTX resource script lacks its RCDATA registration")
endif()
