# cmake for OpenCV2 Cookbook
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# define the project name
set(project_name "Chapter09")

# set the project namee
project("${project_name}")

# add opencv package to the project
find_package( OpenCV REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")

# add opencv include directories to the project
include_directories( ${OpenCV_INCLUDE_DIRS} ) 

# add executable
add_executable( patches patches.cpp)
add_executable( matcher matcher.cpp)
add_executable( binaryDescriptors binaryDescriptors.cpp)

# link libraries
target_link_libraries( patches ${OpenCV_LIBS})
target_link_libraries( matcher ${OpenCV_LIBS})
target_link_libraries( binaryDescriptors ${OpenCV_LIBS})

