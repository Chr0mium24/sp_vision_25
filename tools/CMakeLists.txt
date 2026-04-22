cmake_minimum_required(VERSION 3.16)

add_subdirectory(runtime)
add_subdirectory(math)
add_subdirectory(vision)
add_subdirectory(protocol)

add_library(tools STATIC)
target_sources(
    tools PRIVATE
    $<TARGET_OBJECTS:tools_runtime>
    $<TARGET_OBJECTS:tools_math>
    $<TARGET_OBJECTS:tools_vision>
    $<TARGET_OBJECTS:tools_protocol>
)

set_target_properties(
    tools_runtime tools_math tools_vision tools_protocol tools
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
