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

if(APPLE)
    set(SERIAL_BINDING_SOURCES
        ${PROJECT_SOURCE_DIR}/io/serial/src/serial.cc
        ${PROJECT_SOURCE_DIR}/io/serial/src/impl/unix.cc
        ${PROJECT_SOURCE_DIR}/io/serial/src/impl/list_ports/list_ports_osx.cc)
elseif(UNIX)
    set(SERIAL_BINDING_SOURCES
        ${PROJECT_SOURCE_DIR}/io/serial/src/serial.cc
        ${PROJECT_SOURCE_DIR}/io/serial/src/impl/unix.cc
        ${PROJECT_SOURCE_DIR}/io/serial/src/impl/list_ports/list_ports_linux.cc)
else()
    set(SERIAL_BINDING_SOURCES
        ${PROJECT_SOURCE_DIR}/io/serial/src/serial.cc
        ${PROJECT_SOURCE_DIR}/io/serial/src/impl/win.cc
        ${PROJECT_SOURCE_DIR}/io/serial/src/impl/list_ports/list_ports_win.cc)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(CAMERA_VENDOR_LIB_DIR ${PROJECT_SOURCE_DIR}/io/hikrobot/lib/amd64)
  set(MINDVISION_VENDOR_LIB_DIR ${PROJECT_SOURCE_DIR}/io/mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CAMERA_VENDOR_LIB_DIR ${PROJECT_SOURCE_DIR}/io/hikrobot/lib/arm64)
  set(MINDVISION_VENDOR_LIB_DIR ${PROJECT_SOURCE_DIR}/io/mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture for bindings: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

add_library(sp_vision_bindings MODULE
    module.cpp
    io/command_bindings.cpp
    io/camera_bindings.cpp
    io/cboard_bindings.cpp
    io/gimbal_bindings.cpp
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
    ${PROJECT_SOURCE_DIR}/io/hikrobot/hikrobot.cpp
    ${PROJECT_SOURCE_DIR}/io/mindvision/mindvision.cpp
    ${PROJECT_SOURCE_DIR}/io/usbcamera/usbcamera.cpp
    ${PROJECT_SOURCE_DIR}/io/camera.cpp
    ${PROJECT_SOURCE_DIR}/io/cboard.cpp
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
    ${PROJECT_SOURCE_DIR}/io/gimbal/gimbal.cpp
    ${SERIAL_BINDING_SOURCES}
)

target_include_directories(sp_vision_bindings PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/io/serial/include"
    "${PROJECT_SOURCE_DIR}/io/hikrobot/include"
    "${PROJECT_SOURCE_DIR}/io/mindvision/include"
    "${PYBIND11_INCLUDE_DIR}"
)
target_link_directories(sp_vision_bindings PRIVATE
    "${CAMERA_VENDOR_LIB_DIR}"
    "${MINDVISION_VENDOR_LIB_DIR}"
)
target_link_libraries(
    sp_vision_bindings
    PRIVATE Python3::Module yaml-cpp fmt::fmt spdlog::spdlog ${OpenCV_LIBS} openvino::runtime
            MvCameraControl MVSDK usb-1.0)
if(APPLE)
    target_link_libraries(sp_vision_bindings PRIVATE ${FOUNDATION_LIBRARY} ${IOKIT_LIBRARY})
elseif(UNIX)
    target_link_libraries(sp_vision_bindings PRIVATE rt pthread)
endif()
set_target_properties(sp_vision_bindings PROPERTIES
    PREFIX ""
    OUTPUT_NAME "sp_vision_bindings"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)
