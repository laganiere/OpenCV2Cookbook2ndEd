# cmake for OpenCV2 Cookbook
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# define the project name
set(project_name "Chapter05")

# set the project namee
project("${project_name}")

# add opencv package to the project
find_package( OpenCV REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")

# add opencv include directories to the project
include_directories( ${OpenCV_INCLUDE_DIRS} ) 

# add executable
add_executable( morphology morphology.cpp)
add_executable( morpho2 morpho2.cpp)
add_executable( mserFeatures mserFeatures.cpp)
add_executable( segment segment.cpp)

# link libraries
target_link_libraries( morphology ${OpenCV_LIBS})
target_link_libraries( morpho2 ${OpenCV_LIBS})
target_link_libraries( mserFeatures ${OpenCV_LIBS})
target_link_libraries( segment ${OpenCV_LIBS})

