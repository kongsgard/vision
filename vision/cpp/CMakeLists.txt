cmake_minimum_required(VERSION 3.21.0)

set(This Vision)

project(${This} VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
include(GoogleTest)
enable_testing()

add_subdirectory(eigen)
add_subdirectory(googletest)

set(Headers
    Image.hpp
)

set(Sources
    Image.cpp
)

add_library(${This} STATIC ${Sources} ${Headers})
target_link_libraries (${This} Eigen3::Eigen)

add_executable(Main main.cpp)
target_link_libraries(Main PUBLIC
    ${This}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory(tests)
