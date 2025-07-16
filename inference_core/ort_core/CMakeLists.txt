cmake_minimum_required(VERSION 3.8)
project(ort_core)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)


if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(STATUS "Building OnnxRuntime-Core on x86_64 platform ...")
  set(ONNXRUNTIME_DIRS onnxruntime-linux-x64)
  FILE(GLOB ONNXRUNTIME_LIBS onnxruntime-linux-x64/lib/*.so)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building OnnxRuntime-Core on aarch64 platform ...")
  set(ONNXRUNTIME_DIRS onnxruntime-linux-aarch64)
  FILE(GLOB ONNXRUNTIME_LIBS onnxruntime-linux-aarch64/lib/*.so)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. \
                      Supported architectures are x86_64 and aarch64.")
endif()


set(source_file
  src/ort_core.cpp
  src/ort_core_factory.cpp
)

include_directories(
  include
  ${ONNXRUNTIME_DIRS}/include
)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${ONNXRUNTIME_LIBS}
  deploy_core
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.hpp")

install(DIRECTORY ${ONNXRUNTIME_DIRS}/lib/
        DESTINATION lib)
install(DIRECTORY ${ONNXRUNTIME_DIRS}/include/
        DESTINATION include)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)