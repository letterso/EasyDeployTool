cmake_minimum_required(VERSION 3.0.2)
project(rknn_core)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

set(source_file src/rknn_core.cpp
                src/rknn_core_factory.cpp)

add_library(${PROJECT_NAME} SHARED ${source_file})

include_directories(
  include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  deploy_core
  rknnrt
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.hpp")

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
