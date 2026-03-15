cmake_minimum_required(VERSION 3.16.3)

project(sp_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})


add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

####################src###################
add_executable(standard src/standard.cpp)
add_executable(mt_standard src/mt_standard.cpp)
add_executable(standard_mpc src/standard_mpc.cpp)
add_executable(auto_aim_debug_mpc src/auto_aim_debug_mpc.cpp)
add_executable(mt_auto_aim_debug src/mt_auto_aim_debug.cpp)
add_executable(auto_buff_debug src/auto_buff_debug.cpp)
add_executable(auto_buff_debug_mpc src/auto_buff_debug_mpc.cpp)
add_executable(uav src/uav.cpp)
add_executable(uav_debug src/uav_debug.cpp)

target_link_libraries(standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(mt_standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(standard_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(auto_aim_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(mt_auto_aim_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(auto_buff_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(auto_buff_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(uav ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(uav_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

##################calibration################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)

##################tests##################
add_executable(auto_aim_test tests/auto_aim/auto_aim_test.cpp)
add_executable(auto_power_rune_test tests/auto_buff/auto_power_rune_test.cpp)
add_executable(camera_detect_test tests/camera/camera_detect_test.cpp)
add_executable(camera_test tests/camera/camera_test.cpp)
add_executable(camera_save_test tests/camera/camera_save_test.cpp)
add_executable(camera_thread_test tests/camera/camera_thread_test.cpp)
add_executable(camera_window_test tests/camera/camera_window_test.cpp)
add_executable(cboard_test tests/system/cboard_test.cpp)
add_executable(fire_test tests/auto_aim/fire_test.cpp)
add_executable(detector_video_test tests/auto_aim/detector_video_test.cpp)
add_executable(gimbal_response_test tests/gimbal/gimbal_response_test.cpp)
add_executable(multi_usbcamera_test tests/camera/multi_usbcamera_test.cpp)
add_executable(usbcamera_detect_test tests/camera/usbcamera_detect_test.cpp)
add_executable(usbcamera_test tests/camera/usbcamera_test.cpp)
add_executable(handeye_test tests/camera/handeye_test.cpp)
add_executable(dm_test tests/system/dm_test.cpp)


target_link_libraries(auto_aim_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(auto_power_rune_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io)
target_link_libraries(camera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim io)
target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(camera_save_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(camera_thread_test ${OpenCV_LIBS} fmt::fmt auto_aim tools io)
target_link_libraries(camera_window_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(cboard_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(fire_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(detector_video_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim)
target_link_libraries(gimbal_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(multi_usbcamera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(usbcamera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(usbcamera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(handeye_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(dm_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)


add_executable(gimbal_test tests/gimbal/gimbal_test.cpp)
target_link_libraries(gimbal_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test tests/planner/planner_test.cpp)
target_link_libraries(planner_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test_offline tests/planner/planner_test_offline.cpp)
target_link_libraries(planner_test_offline ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

##################diagnostics##################
add_executable(auto_aim_ui_test diagnostics/auto_aim/auto_aim_ui_test.cpp)
add_executable(auto_aim_ui_tune diagnostics/auto_aim/auto_aim_ui_tune.cpp)
target_link_libraries(auto_aim_ui_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(auto_aim_ui_tune ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)

add_executable(gimbal_ui_test diagnostics/gimbal/gimbal_ui_test.cpp)
target_link_libraries(gimbal_ui_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)

add_executable(hero_gimbal_ui_test diagnostics/gimbal/hero_gimbal_ui_test.cpp)
target_link_libraries(hero_gimbal_ui_test ${OpenCV_LIBS} fmt::fmt tools io)

add_executable(usb_aim_rx_test diagnostics/gimbal/usb_aim_rx_test.cpp)
target_link_libraries(usb_aim_rx_test ${OpenCV_LIBS} fmt::fmt tools io)

add_executable(gimbal_serial_probe diagnostics/gimbal/gimbal_serial_probe.cpp)
target_link_libraries(gimbal_serial_probe ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)

add_executable(gimbal_link_diag_test diagnostics/gimbal/gimbal_link_diag_test.cpp)
target_link_libraries(gimbal_link_diag_test ${OpenCV_LIBS} fmt::fmt tools io)

function(set_test_output target group)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests/${group}"
    )
endfunction()

function(set_diag_output target group)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/diag/${group}"
    )
endfunction()

set_test_output(auto_aim_test auto_aim)
set_test_output(detector_video_test auto_aim)
set_test_output(fire_test auto_aim)

set_test_output(auto_power_rune_test auto_buff)

set_test_output(camera_detect_test camera)
set_test_output(camera_test camera)
set_test_output(camera_save_test camera)
set_test_output(camera_thread_test camera)
set_test_output(camera_window_test camera)
set_test_output(usbcamera_detect_test camera)
set_test_output(usbcamera_test camera)
set_test_output(multi_usbcamera_test camera)
set_test_output(handeye_test camera)

set_test_output(gimbal_test gimbal)
set_test_output(gimbal_response_test gimbal)

set_test_output(planner_test planner)
set_test_output(planner_test_offline planner)

set_test_output(cboard_test system)
set_test_output(dm_test system)

set_diag_output(auto_aim_ui_test auto_aim)
set_diag_output(auto_aim_ui_tune auto_aim)
set_diag_output(gimbal_ui_test gimbal)
set_diag_output(hero_gimbal_ui_test gimbal)
set_diag_output(usb_aim_rx_test gimbal)
set_diag_output(gimbal_serial_probe gimbal)
set_diag_output(gimbal_link_diag_test gimbal)

# 检查 ROS 环境
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(sensor_msgs QUIET)
find_package(cv_bridge QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)



# 只有在找到 ROS 环境时才会添加 sentry 和 publish_test
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
    # 编译 sentry 和 publish_test
    add_executable(sentry src/sentry.cpp)
    add_executable(sentry_bp src/sentry_bp.cpp)
    add_executable(sentry_debug src/sentry_debug.cpp)
    add_executable(sentry_multithread src/sentry_multithread.cpp)
    add_executable(publish_test tests/ros2/publish_test.cpp)
    add_executable(subscribe_test tests/ros2/subscribe_test.cpp)
    add_executable(topic_loop_test tests/ros2/topic_loop_test.cpp)

    target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_bp ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_multithread ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim)
    target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
    target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)

    set_test_output(publish_test ros2)
    set_test_output(subscribe_test ros2)
    set_test_output(topic_loop_test ros2)
else()
    message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
endif()
