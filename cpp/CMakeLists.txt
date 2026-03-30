cmake_minimum_required(VERSION 3.20)
project(mocap_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(mocap_core
    bindings/mocap_bindings.cpp
    triangulation/triangulator.cpp
)

target_include_directories(mocap_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/triangulation
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(mocap_core PRIVATE Eigen3::Eigen)

install(TARGETS mocap_core DESTINATION .)
