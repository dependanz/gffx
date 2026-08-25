get_filename_component(copy_directory "${PLUGIN_COPY}" DIRECTORY)
file(MAKE_DIRECTORY "${copy_directory}")
file(COPY_FILE "${PLUGIN_SOURCE}" "${PLUGIN_COPY}" ONLY_IF_DIFFERENT)

execute_process(
    COMMAND "${TEST_PROGRAM}" "${PLUGIN_COPY}" loaded compatible
    RESULT_VARIABLE loaded_result
)
if(NOT loaded_result EQUAL 0)
    message(FATAL_ERROR "copied synthetic plugin did not load: ${loaded_result}")
endif()

file(REMOVE "${PLUGIN_COPY}")
if(EXISTS "${PLUGIN_COPY}")
    message(FATAL_ERROR "test plugin copy could not be removed")
endif()

execute_process(
    COMMAND "${TEST_PROGRAM}" "${PLUGIN_COPY}" "not found" absent
    RESULT_VARIABLE removed_result
)
if(NOT removed_result EQUAL 0)
    message(FATAL_ERROR
        "CPU capability probing did not survive plugin removal: ${removed_result}"
    )
endif()
