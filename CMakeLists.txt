# Run in terminal
# mkdir build
# cd build
# conan install .. --output-folder=build/generated --build=missing
# cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=build/generated/conan_toolchain.cmake
# cmake --build build --config Release

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hierarchical_sentence_transformer)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/generated/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/generated/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/generated/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the Conan-generated configurations
include(${CMAKE_BINARY_DIR}/generated/conan_toolchain.cmake)

# Find packages
find_package(Catch2 3 REQUIRED)
find_package(Torch REQUIRED)

# Add source files
set(SOURCES
    token_handler.cpp
    hierarchical_sentence_transformer.cpp
    main.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} Catch2::Catch2WithMain)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})

# Add CUDA compile definitions if CUDA is available
if(TORCH_CUDA_AVAILABLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_CUDA)
endif()

# Set the startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Add Catch2 specific configuration
include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME})