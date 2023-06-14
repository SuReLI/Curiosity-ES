cmake_minimum_required(VERSION 2.8.3)

project(Mazes)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
set(CMAKE_BUILD_TYPE Debug)


# Eigen
#find_package(Eigen3)

# Pybind11
add_subdirectory(pybind11)
# Include
include_directories(include)

# Vibes
add_library(Maze src/Maze.cpp)
add_library(Agent src/Agent.cpp)
add_library(Block src/Block.cpp)
add_library(utils src/utils.cpp)
add_library(vibes src/vibes.cpp)



pybind11_add_module(Game src/main.cpp  )
target_link_libraries(Game Maze Agent Block utils vibes )

