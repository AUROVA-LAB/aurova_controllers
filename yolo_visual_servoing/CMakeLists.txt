cmake_minimum_required(VERSION 3.0.2)
project(yolo_visual_servoing)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS "-std=gnu++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  detection_msgs
  yolact_ros_msgs
  cv_bridge
  eigen_conversions
  image_transport
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)

add_message_files(
	FILES
	selectImg.msg
)

generate_messages(
	DEPENDENCIES
	std_msgs
)

catkin_package(
CATKIN_DEPENDS #detection_msgs roscpp std_msgs message_runtime

)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable(yoloVisualServo_node src/yoloVisualServo_node.cpp)
add_executable(switchImage_node src/imageRGBTopic_switch.cpp)
add_executable(switchDepth_node src/imageDepthTopic_switch.cpp)
add_executable(depth_measure_node src/depth_measure.cpp)
add_executable(depth_measure_segmentation_node src/depth_measure_segmentation.cpp)

target_link_libraries(yoloVisualServo_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
${Boost_LIBRARIES})

target_link_libraries(switchImage_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
${Boost_LIBRARIES})

target_link_libraries(switchDepth_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
${Boost_LIBRARIES})

target_link_libraries(depth_measure_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
${Boost_LIBRARIES})

target_link_libraries(depth_measure_segmentation_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
${Boost_LIBRARIES})






