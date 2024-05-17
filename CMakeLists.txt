cmake_minimum_required(VERSION 3.28)
project(FFT)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined")

enable_testing()
add_subdirectory(googletest)

add_executable(FFT tests.cpp
        FastFourierTransform.cpp
        FastFourierTransform.h
        FastFourierTransform.cpp
        FastFourierTransform.h)
target_link_libraries(FFT gtest_main gmock_main)
