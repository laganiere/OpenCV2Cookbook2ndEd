# cmake for OpenCV2 Cookbook
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# define the project name
set(project_name "Chapter10")

# set the project namee
project("${project_name}")

# add opencv package to the project
find_package( OpenCV REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")

# add opencv include directories to the project
include_directories( ${OpenCV_INCLUDE_DIRS} ) 
# add include directory
include_directories (${Chapter10_SOURCE_DIR}) 

# add executable
add_executable( calibrate CameraCalibrator.cpp calibrate.cpp)
add_executable( estimateF estimateF.cpp)
add_executable( robustmatching robustmatching.cpp)
add_executable( estimateH estimateH.cpp)

# link libraries
target_link_libraries( calibrate ${OpenCV_LIBS})
target_link_libraries( estimateF ${OpenCV_LIBS})
target_link_libraries( robustmatching ${OpenCV_LIBS})
target_link_libraries( estimateH ${OpenCV_LIBS})

