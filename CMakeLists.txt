cmake_minimum_required(VERSION 3.20)
project(RiskGuard VERSION 2.0.0 LANGUAGES CXX)

# ════════════════════════════════════════════════
#  Tận dụng: std::span, Concepts, std::format
# ════════════════════════════════════════════════
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kiểm tra compiler hỗ trợ C++20
if(MSVC)
    if(MSVC_VERSION LESS 1929)
        message(FATAL_ERROR "MSVC 2019 v16.9+ required for C++20 support")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
        message(FATAL_ERROR "GCC 11+ required for C++20 support")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0")
        message(FATAL_ERROR "Clang 13+ required for C++20 support")
    endif()
endif()

# ════════════════════════════════════════════════
#  Build Options
# ════════════════════════════════════════════════
option(RISKGUARD_BUILD_TESTS  "Build unit test runner"      ON)
option(RISKGUARD_ENABLE_ASAN  "Enable AddressSanitizer"     OFF)

# ════════════════════════════════════════════════
#  Source Files — Core + Layers + Loss + Utils
# ════════════════════════════════════════════════
set(RISKGUARD_SOURCES
    # Core
    src/core/Matrix.cpp

    # Layers
    src/layers/LinearLayer.cpp
    src/layers/SigmoidLayer.cpp

    # Network
    src/NeuralNetwork.cpp

    # Utils — Data Pipeline & Tooling
    src/utils/Dashboard.cpp
    src/utils/DataLoader.cpp
)

# ════════════════════════════════════════════════
#  Thư viện tĩnh cốt lõi
#  (dùng chung cho app/ và tests/)
# ════════════════════════════════════════════════
add_library(riskguard_lib STATIC ${RISKGUARD_SOURCES})

target_include_directories(riskguard_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Cờ biên dịch phân nhánh theo compiler
target_compile_options(riskguard_lib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall -Wextra -Wpedantic -Wno-unused-parameter
        -O2 -march=native
        -static -static-libgcc -static-libstdc++
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /O2 /permissive-
        /wd4100   # unreferenced formal parameter
    >
)

# AddressSanitizer (debug mode)
if(RISKGUARD_ENABLE_ASAN)
    target_compile_options(riskguard_lib PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(riskguard_lib PUBLIC -fsanitize=address)
endif()

# ════════════════════════════════════════════════
#  Executable chính — Full credit risk pipeline
# ════════════════════════════════════════════════
# app/main.cpp does not exist yet, we will just use src/main.cpp which is a test runner right now
add_executable(riskguard app/main.cpp)
target_link_libraries(riskguard PRIVATE riskguard_lib)

# ════════════════════════════════════════════════
#  Test Runner — Tự viết, không dùng thư viện ngoài
# ════════════════════════════════════════════════
if(RISKGUARD_BUILD_TESTS)
    enable_testing()

    set(RISKGUARD_TEST_SOURCES
        src/main.cpp
    )

    add_executable(riskguard_tests ${RISKGUARD_TEST_SOURCES})
    target_link_libraries(riskguard_tests PRIVATE riskguard_lib)
    target_compile_definitions(riskguard_tests PRIVATE
        RISKGUARD_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/raw"
    )

    add_test(NAME AllTests COMMAND riskguard_tests)
endif()

# ════════════════════════════════════════════════
#  Thông tin build khi cmake configure
# ════════════════════════════════════════════════
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  RiskGuard ML Framework v${PROJECT_VERSION}")
message(STATUS "  C++ Standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Tests  : ${RISKGUARD_BUILD_TESTS}")
message(STATUS "  ASAN         : ${RISKGUARD_ENABLE_ASAN}")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
