cmake_minimum_required(VERSION 3.16)
project(neural_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(neural_core STATIC
    src/core.cpp
)

target_include_directories(neural_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(cpp_neural
    src/main.cpp
)
target_link_libraries(cpp_neural PRIVATE neural_core)

include(CTest)
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(run_tests tests/tests.cpp)
    target_link_libraries(run_tests PRIVATE neural_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(run_tests)
endif()
