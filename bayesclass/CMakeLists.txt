cmake_minimum_required(VERSION 3.20)
project(feature)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

add_executable(feature FeatureSelect.cpp)
