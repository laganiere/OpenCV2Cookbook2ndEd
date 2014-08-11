# cmake for OpenCV2 Cookbook
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# define the project name
set(project_name "Chapter02")

# set the project namee
project("${project_name}")

# add opencv package to the project
find_package( OpenCV REQUIRED )
MESSAGE(“OpenCV version : ${OpenCV_VERSION}”)

# add opencv include directories to the project
include_directories( ${OpenCV_INCLUDE_DIRS} ) 

# add executable
add_executable( saltImage saltImage.cpp)
add_executable( colorReduce colorReduce.cpp)
add_executable( contrast contrast.cpp)
add_executable( addImages addImages.cpp)
add_executable( remapping remapping.cpp)

# link libraries
target_link_libraries( saltImage ${OpenCV_LIBS})
target_link_libraries( colorReduce ${OpenCV_LIBS})
target_link_libraries( contrast ${OpenCV_LIBS})
target_link_libraries( addImages ${OpenCV_LIBS})
target_link_libraries( remapping ${OpenCV_LIBS})
