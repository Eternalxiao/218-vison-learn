cmake_minimum_required(VERSION 3.16)
project(paper_assembler VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Core library (header-only in practice, but compiled for testing) ---
add_library(puzzle_core STATIC
    core/geometry.cpp
    core/solver.cpp
)
target_include_directories(puzzle_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(puzzle_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3 -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast>
)

# --- C ABI shared library ---
add_library(paper_assembler SHARED
    capi/assembler.cpp
)
target_link_libraries(paper_assembler PRIVATE puzzle_core)
target_include_directories(paper_assembler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# On Windows: paper_assembler.dll; on Linux/MaixCAM: libpaper_assembler.so
set_target_properties(paper_assembler PROPERTIES
    OUTPUT_NAME paper_assembler
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(paper_assembler PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3 -ffast-math -fvisibility=hidden>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast>
)

# Linker: export only C ABI symbols
if(UNIX)
    target_link_options(paper_assembler PRIVATE
        -Wl,--exclude-libs,ALL
        -Wl,-soname,libpaper_assembler.so
    )
endif()

if(MINGW)
    target_link_options(paper_assembler PRIVATE -static-libgcc -static-libstdc++)
endif()

# --- Standalone test executable (PC only) ---
option(BUILD_TESTS "Build test binary" ON)
if(BUILD_TESTS)
    add_executable(test_puzzle
        tests/test_main.cpp
    )
    target_link_libraries(test_puzzle PRIVATE puzzle_core)
    target_include_directories(test_puzzle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Install
install(TARGETS paper_assembler puzzle_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
