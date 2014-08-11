# cmake for OpenCV2 Cookbook
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# define the project name
set(project_name "Chapter11")

# set the project namee
project("${project_name}")

# add opencv package to the project
find_package( OpenCV REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")

# add opencv include directories to the project
include_directories( ${OpenCV_INCLUDE_DIRS} ) 
# add include directory
include_directories (${Chapter11_SOURCE_DIR}) 

# add executable
add_executable( videoprocessing videoprocessing.cpp)
add_executable( tracking tracking.cpp)
add_executable( foreground foreground.cpp)

# link libraries
target_link_libraries( videoprocessing ${OpenCV_LIBS})
target_link_libraries( tracking ${OpenCV_LIBS})
target_link_libraries( foreground ${OpenCV_LIBS})

