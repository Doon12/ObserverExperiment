cmake_minimum_required(VERSION 2.8.4)
project(myproject)

set(CMAKE_SUPPRESS_REGENERATION TRUE)

add_subdirectory(/home/nvidia/vectornav/cpp libvncxx)
add_subdirectory(/home/nvidia/RTIMULib2/RTIMULib RTIMULib)
MESSAGE("${CMAKE_CXX_COMPILER_ID}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++11")
endif()

set(SOURCE_FILES
	main.cpp)

find_package(Threads)
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(/home/nvidia/aruco-markers/pose_estimation/include)
include_directories(/home/nvidia/vectornav/cpp/include)
include_directories(/home/nvidia/RTIMULib2/RTIMULib)

link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(myproject ${SOURCE_FILES})

target_link_libraries(myproject PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(myproject PRIVATE ${OpenCV_LIBRARIES})
target_link_libraries(myproject PRIVATE libvncxx)
target_link_libraries(myproject PRIVATE RTIMULib)

if (UNIX OR APPLE)
	target_link_libraries(myproject LINK_PUBLIC pthread)
else()
	target_link_libraries(myproject LINK_PUBLIC Advapi32 SetupAPI)
endif()
