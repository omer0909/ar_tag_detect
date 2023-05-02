cmake_minimum_required(VERSION 3.8)
project(opencv_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(opencv_demo REQUIRED) 

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

add_executable(image_process_node src/image_process.cpp)
ament_target_dependencies(image_process_node rclcpp std_msgs sensor_msgs cv_bridge image_transport OpenCV geometry_msgs opencv_demo)

add_executable(image_process_test_node src/image_process_test.cpp)
ament_target_dependencies(image_process_test_node rclcpp std_msgs sensor_msgs image_transport geometry_msgs opencv_demo tf2 tf2_ros)


install(TARGETS
   image_process_node
   image_process_test_node
   DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY src
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArInfo.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)


ament_package()
