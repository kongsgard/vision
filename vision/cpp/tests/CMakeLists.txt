cmake_minimum_required(VERSION 3.21.0)

set(This Tests)

set(Sources
    ImageTests.cpp
)

add_executable(${This} ${Sources})
target_link_libraries(${This} PUBLIC
    gtest_main
    Vision
)

gtest_discover_tests(${This})
