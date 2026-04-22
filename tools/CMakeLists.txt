cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT 
    runtime/exiter.cpp
    math/extended_kalman_filter.cpp
    math/ransac_sine_fitter.cpp
    vision/img_tools.cpp
    math/math_tools.cpp
    runtime/plotter.cpp
    math/trajectory.cpp
    runtime/recorder.cpp
    runtime/logger.cpp
    math/pid.cpp
    protocol/crc.cpp
)
