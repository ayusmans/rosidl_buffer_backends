# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(libtorch_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

set(LIBTORCH_VERSION "2.11.0" CACHE STRING "LibTorch version to download")
set(LIBTORCH_CUDA_VERSION "" CACHE STRING
  "CUDA variant override (e.g. cu118, cu124, cu126, cpu). Auto-detected if empty.")

option(FORCE_BUILD_VENDOR_PKG
  "Build vendor packages from source, even if system-installed packages are available"
  OFF)

if(LIBTORCH_CUDA_VERSION STREQUAL "")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    # cu126 matches modern CUDA 12.x drivers (backward-compatible).
    set(LIBTORCH_CUDA_VERSION "cu126")
    message(STATUS "${PROJECT_NAME}: CUDA ${CUDAToolkit_VERSION} detected, using cu126")
  else()
    set(LIBTORCH_CUDA_VERSION "cpu")
    message(STATUS "${PROJECT_NAME}: no CUDA toolkit found, using CPU build")
  endif()
endif()

find_package(Torch QUIET)

if(NOT Torch_FOUND OR FORCE_BUILD_VENDOR_PKG)
  if(Torch_FOUND)
    message(STATUS "${PROJECT_NAME}: forcing vendor build (FORCE_BUILD_VENDOR_PKG)")
  endif()

  # Starting with LibTorch 2.8.0, PyTorch dropped the "-cxx11-abi-" infix from
  # the Linux archive name (they no longer ship the pre-cxx11-abi variant).
  if(LIBTORCH_VERSION VERSION_GREATER_EQUAL "2.8.0")
    set(_archive_prefix "libtorch-shared-with-deps")
  else()
    set(_archive_prefix "libtorch-cxx11-abi-shared-with-deps")
  endif()
  set(_url "https://download.pytorch.org/libtorch/${LIBTORCH_CUDA_VERSION}/${_archive_prefix}-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA_VERSION}.zip")
  set(_archive "${CMAKE_CURRENT_BINARY_DIR}/libtorch.zip")
  set(_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/libtorch_extracted")

  if(NOT EXISTS "${_extract_dir}/libtorch/share/cmake/Torch/TorchConfig.cmake")
    message(STATUS "${PROJECT_NAME}: downloading LibTorch ${LIBTORCH_VERSION}+${LIBTORCH_CUDA_VERSION}")
    message(STATUS "${PROJECT_NAME}: URL: ${_url}")
    file(DOWNLOAD "${_url}" "${_archive}" SHOW_PROGRESS STATUS _dl_status)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      list(GET _dl_status 1 _dl_msg)
      message(FATAL_ERROR "${PROJECT_NAME}: download failed: ${_dl_msg}")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_extract_dir}")
    file(REMOVE "${_archive}")
  else()
    message(STATUS "${PROJECT_NAME}: using cached LibTorch at ${_extract_dir}")
  endif()

  install(DIRECTORY "${_extract_dir}/libtorch/"
    DESTINATION "opt/${PROJECT_NAME}"
    USE_SOURCE_PERMISSIONS
  )

  ament_index_register_resource("vendor_packages" CONTENT "opt/${PROJECT_NAME}")

  # Install environment hooks so downstream binaries can locate the vendored
  # shared libraries at runtime (LD_LIBRARY_PATH / PATH).
  if(APPLE)
    set(_LIBDIR_VAR_NAME "DYLD_LIBRARY_PATH")
  elseif(WIN32)
    set(_LIBDIR_VAR_NAME "PATH")
  else()
    set(_LIBDIR_VAR_NAME "LD_LIBRARY_PATH")
  endif()
  ament_environment_hooks(
    "${ament_cmake_vendor_package_DIR}/templates/vendor_package.sh.in"
    "${ament_cmake_vendor_package_DIR}/templates/vendor_package.dsv.in"
  )
else()
  message(STATUS "${PROJECT_NAME}: system LibTorch ${Torch_VERSION} found, skipping download")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(
  CONFIG_EXTRAS_POST "libtorch_vendor-extras.cmake.in"
)
