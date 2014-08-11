# cmake for OpenCV2 Cookbook
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# define the project name
set(project_name "Chapter04")

# set the project namee
project("${project_name}")

# add opencv package to the project
find_package( OpenCV REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")

# add opencv include directories to the project
include_directories( ${OpenCV_INCLUDE_DIRS} ) 

# add executable
add_executable( histograms histograms.cpp)
add_executable( contentfinder contentfinder.cpp histogram.cpp)
add_executable( retrieve retrieve.cpp)
add_executable( tracking tracking.cpp)

# link libraries
target_link_libraries( histograms ${OpenCV_LIBS})
target_link_libraries( contentfinder ${OpenCV_LIBS})
target_link_libraries( retrieve ${OpenCV_LIBS})
target_link_libraries( tracking ${OpenCV_LIBS})

