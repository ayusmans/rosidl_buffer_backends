cmake_minimum_required(VERSION 3.20)
project(cuda_buffer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(cuda_buffer_backend_msgs REQUIRED)
find_package(rmw REQUIRED)
find_package(rcutils REQUIRED)

find_package(CUDAToolkit REQUIRED)
# Reads the NvSci endpoint pool inventory; header-only.
find_package(nlohmann_json 3 QUIET)
message(STATUS "${PROJECT_NAME}: CUDA ${CUDAToolkit_VERSION} from ${CUDAToolkit_BIN_DIR}")

add_library(${PROJECT_NAME} SHARED
  src/cuda_buffer.cpp
  src/cuda_buffer_impl.cpp
  src/cuda_buffer_ipc_manager.cpp
  src/cuda_memory_pool.cpp
  src/deferred_release.cpp
  src/host_endpoint_manager.cpp
  src/vmm_buffer.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    rosidl_buffer::rosidl_buffer
    rmw::rmw
    rcutils::rcutils
    ${cuda_buffer_backend_msgs_TARGETS}
  PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    rt
)

# NvSci storage backend. Built when the NvSci headers and libraries are found
# (Jetson, or the x86 "nvsci" package); set CUDA_BUFFER_NVSCI=ON to require it
# or OFF to leave it out. The definition is PUBLIC because the plugin compiles
# its NvSci transport only when it is set. Whether NvSci is usable is decided
# at runtime: a build with NvSci still runs where NvSci is not available.
set(CUDA_BUFFER_NVSCI "AUTO" CACHE STRING "Build the NvSci storage backend: ON, OFF or AUTO")
set_property(CACHE CUDA_BUFFER_NVSCI PROPERTY STRINGS ON OFF AUTO)
if(NOT CUDA_BUFFER_NVSCI STREQUAL "OFF")
  find_path(NVSCI_INCLUDE_DIR nvscibuf.h PATH_SUFFIXES nvsci)
  find_library(NVSCIBUF_LIBRARY nvscibuf PATH_SUFFIXES nvidia)
  find_library(NVSCISYNC_LIBRARY nvscisync PATH_SUFFIXES nvidia)
  find_library(NVSCIIPC_LIBRARY nvsciipc PATH_SUFFIXES nvidia)
  if(NVSCI_INCLUDE_DIR AND NVSCIBUF_LIBRARY AND NVSCISYNC_LIBRARY AND NVSCIIPC_LIBRARY
      AND nlohmann_json_FOUND)
    message(STATUS "${PROJECT_NAME}: NvSci storage backend enabled (${NVSCI_INCLUDE_DIR})")
    target_sources(${PROJECT_NAME} PRIVATE
      src/nvsci_buffer.cpp
      src/nvsci_endpoint_pool.cpp
      src/nvsci_ledger.cpp
      src/nvsci_peer.cpp
      src/nvsci_runtime.cpp
    )
    target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_BUFFER_ENABLE_NVSCI)
    target_include_directories(${PROJECT_NAME} PUBLIC ${NVSCI_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME}
      PUBLIC ${NVSCIBUF_LIBRARY} ${NVSCISYNC_LIBRARY} ${NVSCIIPC_LIBRARY}
      PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
  elseif(CUDA_BUFFER_NVSCI STREQUAL "ON")
    message(FATAL_ERROR
      "CUDA_BUFFER_NVSCI=ON but the NvSci headers or libraries, or nlohmann_json, were not found")
  else()
    message(STATUS "${PROJECT_NAME}: NvSci not found; NvSci storage backend disabled")
  endif()
endif()

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

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rosidl_buffer cuda_buffer_backend_msgs rmw rcutils)
ament_export_libraries(${PROJECT_NAME})
ament_export_include_directories(include/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_ipc_dispatcher_lifetime
    test/test_ipc_dispatcher_lifetime.cpp
  )
  if(TARGET test_ipc_dispatcher_lifetime)
    target_link_libraries(test_ipc_dispatcher_lifetime ${PROJECT_NAME})
  endif()

  ament_add_gtest(test_cuda_buffer
    test/test_cuda_buffer.cpp
  )
  if(TARGET test_cuda_buffer)
    target_link_libraries(test_cuda_buffer
      ${PROJECT_NAME}
      rosidl_buffer::rosidl_buffer
      CUDA::cudart
      CUDA::cuda_driver
      rt
    )
  endif()

  # The ledger and the endpoint pool are free of NvSci and CUDA, so they are
  # tested on every machine.
  ament_add_gtest(test_nvsci_ledger
    test/test_nvsci_ledger.cpp
    src/nvsci_ledger.cpp
  )
  if(TARGET test_nvsci_ledger)
    target_include_directories(test_nvsci_ledger PRIVATE src)
  endif()

  if(nlohmann_json_FOUND)
    ament_add_gtest(test_nvsci_endpoint_pool
      test/test_nvsci_endpoint_pool.cpp
      src/nvsci_endpoint_pool.cpp
    )
    if(TARGET test_nvsci_endpoint_pool)
      target_include_directories(test_nvsci_endpoint_pool PRIVATE src)
      target_link_libraries(test_nvsci_endpoint_pool nlohmann_json::nlohmann_json rt)
    endif()
  endif()

  ament_add_gtest(test_storage_selection
    test/test_storage_selection.cpp
  )
  if(TARGET test_storage_selection)
    target_link_libraries(test_storage_selection
      ${PROJECT_NAME}
      rosidl_buffer::rosidl_buffer
      CUDA::cudart
    )
  endif()

endif()

ament_package()
