cmake_minimum_required(VERSION 3.16)

add_executable(publish_test publish_test.cpp)
add_executable(subscribe_test subscribe_test.cpp)
add_executable(topic_loop_test topic_loop_test.cpp)

target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim)
target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)

set_test_output(publish_test ros2)
set_test_output(subscribe_test ros2)
set_test_output(topic_loop_test ros2)
