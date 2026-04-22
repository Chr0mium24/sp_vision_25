cmake_minimum_required(VERSION 3.16)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_INCLUDE_RESULT
)

if(NOT PYBIND11_INCLUDE_RESULT EQUAL 0 OR NOT EXISTS "${PYBIND11_INCLUDE_DIR}/pybind11/pybind11.h")
    message(FATAL_ERROR "pybind11 headers not found in the active Python environment.")
endif()

add_library(sp_vision_bindings MODULE
    module.cpp
    tools/crc_bindings.cpp
    tools/ekf_bindings.cpp
    auto_aim/armor_bindings.cpp
    auto_aim/solver_bindings.cpp
    auto_aim/target_bindings.cpp
    auto_aim/tracker_bindings.cpp
    auto_aim/aimer_bindings.cpp
    auto_aim/runtime_bindings.cpp
    ${PROJECT_SOURCE_DIR}/tools/crc.cpp
    ${PROJECT_SOURCE_DIR}/tools/extended_kalman_filter.cpp
    ${PROJECT_SOURCE_DIR}/tools/logger.cpp
    ${PROJECT_SOURCE_DIR}/tools/math_tools.cpp
    ${PROJECT_SOURCE_DIR}/tools/img_tools.cpp
    ${PROJECT_SOURCE_DIR}/tools/trajectory.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/armor.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/solver.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/target.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/tracker.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/aimer.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/auto_aim_runtime.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/yolo.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/classifier.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/detector.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/yolos/yolov5.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/yolos/yolov8.cpp
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim/yolos/yolo11.cpp
)

target_include_directories(sp_vision_bindings PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PYBIND11_INCLUDE_DIR}"
)
target_link_libraries(sp_vision_bindings PRIVATE Python3::Module yaml-cpp fmt::fmt spdlog::spdlog ${OpenCV_LIBS} openvino::runtime)
set_target_properties(sp_vision_bindings PROPERTIES
    PREFIX ""
    OUTPUT_NAME "sp_vision_bindings"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)
