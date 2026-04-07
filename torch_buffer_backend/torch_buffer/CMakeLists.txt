cmake_minimum_required(VERSION 3.8)
project(torch_buffer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(torch_buffer_backend_msgs REQUIRED)
find_package(rmw REQUIRED)
find_package(rcutils REQUIRED)

find_package(cuda_buffer QUIET)
if(cuda_buffer_FOUND)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "torch_buffer: CUDA backend enabled (cuda_buffer found)")
else()
  message(STATUS "torch_buffer: CUDA backend disabled (cuda_buffer not found)")
endif()

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

find_package(Torch REQUIRED)

# Header-only library (no compiled sources)
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  INTERFACE
    ${TORCH_LIBRARIES}
    rosidl_buffer::rosidl_buffer
    rmw::rmw
    rcutils::rcutils
    ${torch_buffer_backend_msgs_TARGETS}
)

if(cuda_buffer_FOUND)
  target_link_libraries(${PROJECT_NAME}
    INTERFACE
      cuda_buffer::cuda_buffer
      CUDA::cudart
  )
endif()

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(${PROJECT_NAME})
ament_export_dependencies(rosidl_buffer torch_buffer_backend_msgs rmw rcutils)
if(cuda_buffer_FOUND)
  ament_export_dependencies(cuda_buffer)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_torch_buffer
    test/test_torch_buffer.cpp
  )
  if(TARGET test_torch_buffer)
    target_link_libraries(test_torch_buffer
      ${PROJECT_NAME}
      ${TORCH_LIBRARIES}
      rosidl_buffer::rosidl_buffer
    )
  endif()
endif()

ament_package()
