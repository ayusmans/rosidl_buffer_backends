cmake_minimum_required(VERSION 3.20)
project(cuda_buffer_examples)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(CUDAToolkit REQUIRED)

ament_auto_add_library(buffer_talker SHARED src/buffer_talker.cpp)
target_link_libraries(buffer_talker CUDA::cudart)
rclcpp_components_register_node(buffer_talker
  PLUGIN "cuda_buffer_examples::BufferTalker"
  EXECUTABLE buffer_talker_node)

ament_auto_add_library(buffer_listener SHARED src/buffer_listener.cpp)
target_link_libraries(buffer_listener CUDA::cudart)
rclcpp_components_register_node(buffer_listener
  PLUGIN "cuda_buffer_examples::BufferListener"
  EXECUTABLE buffer_listener_node)

# The installed executables find the CUDA runtime without LD_LIBRARY_PATH.
set_target_properties(buffer_talker buffer_listener PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_examples_launch.py
    TARGET test_examples_same_process
    ARGS "layout:=same_process"
    TIMEOUT 60
  )
  add_launch_test(test/test_examples_launch.py
    TARGET test_examples_cross_process
    ARGS "layout:=cross_process"
    TIMEOUT 60
  )
endif()

ament_auto_package()
