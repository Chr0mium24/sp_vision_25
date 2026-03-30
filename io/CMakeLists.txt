cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)

add_library(io STATIC
    hikrobot/hikrobot.cpp
    mindvision/mindvision.cpp
    usbcamera/usbcamera.cpp
    camera.cpp
    cboard.cpp
    dm_imu/dm_imu.cpp
    gimbal/gimbal.cpp
)

target_include_directories(io PUBLIC hikrobot/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC hikrobot/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC hikrobot/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_include_directories(io PUBLIC mindvision/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_link_libraries(io MvCameraControl MVSDK usb-1.0 yaml-cpp serial)

find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)

if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND)
  message(STATUS "ROS2 core environment found, enabling gimbal ROS2 transport.")
  target_compile_definitions(io PUBLIC SP_HAS_ROS2_CORE=1)
  ament_target_dependencies(io rclcpp std_msgs)
else()
  message(STATUS "ROS2 core environment not found, building io with serial-only gimbal transport.")
endif()

find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)

if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
  message(STATUS "sp_msgs found, compiling ROS2 navigation glue.")
  target_sources(io PRIVATE ros2/publish2nav.cpp ros2/ros2.cpp ros2/subscribe2nav.cpp)
  ament_target_dependencies(io sp_msgs)
else()
  message(STATUS "sp_msgs ROS2 glue not found, skipping navigation bridge sources.")
endif()
