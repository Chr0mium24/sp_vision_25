cmake_minimum_required(VERSION 3.16)

add_library(
    tools_math OBJECT
    extended_kalman_filter.cpp
    ransac_sine_fitter.cpp
    math_tools.cpp
    trajectory.cpp
    pid.cpp
)
