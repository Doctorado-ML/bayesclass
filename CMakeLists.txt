cmake_minimum_required(VERSION 3.20)
project(feature)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

add_executable(feature bayesclass/cpp/FeatureSelect.cpp)
