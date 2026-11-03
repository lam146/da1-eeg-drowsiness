cmake_minimum_required(VERSION 3.16)
project(n16_eeg_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(n16_runtime
    src/main.cpp
    src/blink.cpp
    src/fft.cpp
    src/model.cpp
    src/runtime.cpp
    src/serial_port.cpp
    src/signal.cpp
    src/thinkgear.cpp
)

target_include_directories(n16_runtime PRIVATE include)

if(MSVC)
    target_compile_options(n16_runtime PRIVATE /W4 /permissive-)
else()
    target_compile_options(n16_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME n16_self_test COMMAND n16_runtime self-test)
