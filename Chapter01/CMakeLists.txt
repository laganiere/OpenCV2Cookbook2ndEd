# cmake for OpenCV2 Cookbook

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# define the project name
set(project_name "Chapter01")

# set the project namee
project("${project_name}")

# add opencv package to the project
find_package( OpenCV REQUIRED )
MESSAGE(“OpenCV version : ${OpenCV_VERSION}”)

# add opencv include directories to the project
include_directories( ${OpenCV_INCLUDE_DIRS} ) 

# add executable
add_executable( loadDisplaySave loadDisplaySave.cpp)
add_executable( logo logo.cpp)
add_executable( mat mat.cpp)

# link libraries
target_link_libraries( loadDisplaySave ${OpenCV_LIBS})
target_link_libraries( logo ${OpenCV_LIBS})
target_link_libraries( mat ${OpenCV_LIBS})
