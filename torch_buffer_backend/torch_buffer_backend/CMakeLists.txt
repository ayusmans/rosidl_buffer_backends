cmake_minimum_required(VERSION 3.8)
project(torch_buffer_backend)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer_backend REQUIRED)
find_package(rosidl_buffer_backend_registry REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(torch_buffer REQUIRED)
find_package(torch_buffer_backend_msgs REQUIRED)
find_package(rcutils REQUIRED)

find_package(CUDAToolkit QUIET)

# Torch cmake setup (Caffe2 needs nvcc + host compiler)
find_program(CUDA_NVCC_EXECUTABLE nvcc
  PATHS /usr/local/cuda/bin ENV CUDA_PATH
)
if(CUDA_NVCC_EXECUTABLE AND NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

set(LIBTORCH_ROOT "$ENV{PIXI_PROJECT_ROOT}/.pixi/envs/default/libtorch")
if(EXISTS "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
  list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
else()
  execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PYTHON_RESULT
  )
  if(TORCH_PYTHON_RESULT EQUAL 0 AND TORCH_CMAKE_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
  endif()
endif()
find_package(Torch REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/torch_buffer_backend_plugin.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${TORCH_LIBRARIES}
  torch_buffer_backend_msgs::torch_buffer_backend_msgs__rosidl_typesupport_fastrtps_cpp
  rosidl_buffer_backend::rosidl_buffer_backend
  rosidl_runtime_cpp::rosidl_runtime_cpp
  rosidl_buffer_backend_registry::rosidl_buffer_backend_registry
  pluginlib::pluginlib
  torch_buffer::torch_buffer
  rcutils::rcutils
)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

install(
  FILES torch_buffer_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(rosidl_buffer_backend torch_buffer_plugin.xml)

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rosidl_buffer_backend
  rosidl_buffer_backend_registry
  rosidl_runtime_cpp
  pluginlib
  torch_buffer
  torch_buffer_backend_msgs
  rcutils
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_add_gtest(test_torch_buffer_backend
    test/test_torch_buffer_backend.cpp
  )
  if(TARGET test_torch_buffer_backend)
    set_target_properties(test_torch_buffer_backend PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE)
    target_link_libraries(test_torch_buffer_backend
      ${PROJECT_NAME}
      ${TORCH_LIBRARIES}
      rosidl_buffer_backend::rosidl_buffer_backend
      rosidl_runtime_cpp::rosidl_runtime_cpp
      rosidl_buffer_backend_registry::rosidl_buffer_backend_registry
      torch_buffer::torch_buffer
      ${sensor_msgs_TARGETS}
    )
    if(CUDAToolkit_FOUND)
      target_link_libraries(test_torch_buffer_backend CUDA::cuda_driver rt)
    endif()
  endif()

  # Publisher component + standalone executable
  add_library(torch_image_publisher_component SHARED
    test/src/torch_image_publisher_node.cpp
  )
  set_target_properties(torch_image_publisher_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(torch_image_publisher_component
    ${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    torch_buffer::torch_buffer
    rclcpp::rclcpp
    rclcpp_components::component
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    rosidl_runtime_cpp::rosidl_runtime_cpp
  )
  if(CUDAToolkit_FOUND)
    target_link_libraries(torch_image_publisher_component CUDA::cuda_driver)
  endif()
  rclcpp_components_register_node(torch_image_publisher_component
    PLUGIN "TorchImagePublisher"
    EXECUTABLE torch_image_publisher_node
  )
  install(TARGETS torch_image_publisher_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  # Subscriber component + standalone executable
  add_library(torch_image_subscriber_component SHARED
    test/src/torch_image_subscriber_node.cpp
  )
  set_target_properties(torch_image_subscriber_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(torch_image_subscriber_component
    ${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    torch_buffer::torch_buffer
    rclcpp::rclcpp
    rclcpp_components::component
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    rosidl_runtime_cpp::rosidl_runtime_cpp
  )
  if(CUDAToolkit_FOUND)
    target_link_libraries(torch_image_subscriber_component CUDA::cuda_driver)
  endif()
  rclcpp_components_register_node(torch_image_subscriber_component
    PLUGIN "TorchImageSubscriber"
    EXECUTABLE torch_image_subscriber_node
  )
  install(TARGETS torch_image_subscriber_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  add_launch_test(test/test_torch_image_inter_pubsub_fastrtps_launch.py
    TARGET test_torch_image_inter_pubsub_fastrtps_launch
    TIMEOUT 30
  )

  add_launch_test(test/test_torch_image_intra_pubsub_fastrtps_launch.py
    TARGET test_torch_image_intra_pubsub_fastrtps_launch
    TIMEOUT 30
  )

  add_launch_test(test/test_torch_image_multi_sub_fastrtps_launch.py
    TARGET test_torch_image_multi_sub_fastrtps_launch
    TIMEOUT 45
  )

  add_launch_test(test/test_torch_image_multi_pub_fastrtps_launch.py
    TARGET test_torch_image_multi_pub_fastrtps_launch
    TIMEOUT 45
  )
endif()

ament_package()
