cmake_minimum_required(VERSION 3.5)

project(laopt
    VERSION 0.1.0
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#### Interface options ####
option(WITH_CASADI "Build Casadi interface" OFF)
option(WITH_IPOPT "Build IPOPT interface" ON)
option(WITH_OSQP "Build OSQP interface" OFF)
option(WITH_PROXQP "Build ProxQP interface" OFF)
option(WITH_QPSWIFT "Build qpSWIFT interface" OFF)
option(WITH_PIQP "Build PIQP interface" ON)
option(WITH_HPIPM "Build HPIPM interface" OFF)

#### Tests/Benchmarks options ####
# Don't build tests and examples if included as subdirectory
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(BUILD_TESTS "Build tests" OFF)
    option(BUILD_EXAMPLES "Build examples" OFF)
else()
    option(BUILD_TESTS "Build tests" ON)
    option(BUILD_EXAMPLES "Build examples" ON)
endif()
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

#### Developer options ####
option(ENABLE_SANITIZERS "Build with sanitizers enabled" OFF)

#### Install options ####
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(ENABLE_INSTALL "Enable install targets" OFF)
else()
    option(ENABLE_INSTALL "Enable install targets" ON)
endif()

# Set build type to RELEASE by default
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()
message(STATUS "Building ${CMAKE_BUILD_TYPE}")

# Set compiler flags according to compiler ID
set(gcc_like_cxx CXX ARMClang AppleClang Clang GNU LCC)
set(msvc_like CXX MSVC)
if (${CMAKE_CXX_COMPILER_ID} IN_LIST gcc_like_cxx)
    list(APPEND compiler_flags -Wall -Wextra "-Wno-unused-parameter" -ftemplate-backtrace-limit=0)
elseif ((${CMAKE_CXX_COMPILER_ID} IN_LIST msvc_like))
    list(APPEND compiler_flags /W4)
endif ()
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Set sanitizer flags
if (ENABLE_SANITIZERS)
    if (${CMAKE_CXX_COMPILER_ID} IN_LIST gcc_like_cxx)
        list(APPEND sanitizer_flags -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    elseif ((${CMAKE_CXX_COMPILER_ID} IN_LIST msvc_like))
        list(APPEND sanitizer_flags /fsanitize=address)
    endif ()
    list(APPEND compiler_flags ${sanitizer_flags})
    message(STATUS "Building with sanitizers: ${sanitizer_flags}")
    unset(sanitizer_flags)
endif ()

# Find Eigen3.4+
if (DEFINED EIGEN3_INCLUDE_DIRS AND NOT TARGET Eigen3::Eigen)
    # Create target for user-defined Eigen 3.3+ path to be used for laOPT
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIRS})
else ()
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif ()
message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIRS}")

add_library(laopt INTERFACE)
target_include_directories(laopt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(laopt INTERFACE Eigen3::Eigen)

# Find Casadi
if (WITH_CASADI)
    find_package(Casadi REQUIRED)
    target_compile_definitions(laopt INTERFACE LAOPT_WITH_CASADI)
    target_link_libraries(laopt INTERFACE casadi)
endif ()

## Find solvers, append to laOPT interface library
# Find IPOPT
if (WITH_IPOPT)
    find_package(Ipopt REQUIRED)
    target_compile_definitions(laopt INTERFACE LAOPT_WITH_IPOPT)
    target_include_directories(laopt INTERFACE ${Ipopt_INCLUDE_DIRS})
    target_link_libraries(laopt INTERFACE ${Ipopt_LIBRARIES})
endif ()

# Find Osqp
if (WITH_OSQP)
    find_package(osqp REQUIRED)
    target_compile_definitions(laopt INTERFACE LAOPT_WITH_OSQP)
    target_link_libraries(laopt INTERFACE osqp::osqp)
endif ()

# Find ProxQP
if (WITH_PROXQP)
    find_package(proxsuite REQUIRED)
    target_compile_definitions(laopt INTERFACE LAOPT_WITH_PROXQP)
    target_link_libraries(laopt INTERFACE proxsuite::proxsuite)
endif ()

# Find qpSwift
if (WITH_QPSWIFT)
    find_package(qpSWIFT REQUIRED)
    target_compile_definitions(laopt INTERFACE LAOPT_WITH_QPSWIFT)
    target_link_libraries(laopt INTERFACE qpSWIFT::qpSWIFT-static)
endif()

# Find PIQP
if (WITH_PIQP)
    find_package(piqp REQUIRED)
    target_compile_definitions(laopt INTERFACE LAOPT_WITH_PIQP)
    target_link_libraries(laopt INTERFACE piqp::piqp)
endif()

# Find HPIPM
if (WITH_HPIPM)
    find_package(blasfeo REQUIRED)
    find_package(hpipm REQUIRED)
    target_compile_definitions(laopt INTERFACE LAOPT_WITH_HPIPM)
    target_link_libraries(laopt INTERFACE blasfeo hpipm)
endif()

target_compile_definitions(laopt INTERFACE -DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # 1 MB - For both Windows and UNIX)
target_compile_features(laopt INTERFACE cxx_std_14)

add_library(laopt::laopt ALIAS laopt)

## Build optional components depending on laOPT library
if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

if (ENABLE_INSTALL)
    include(GNUInstallDirs)

    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
        TARGETS laopt
        EXPORT laoptTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html
    install(
        EXPORT laoptTargets
        FILE laoptTargets.cmake
        NAMESPACE laopt::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/laopt
    )

    include(CMakePackageConfigHelpers)
    configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/laoptConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/laoptConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/laopt
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/laoptConfigVersion.cmake
        VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/laoptConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/laoptConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/laopt
    )

    export(EXPORT laoptTargets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/laoptTargets.cmake
        NAMESPACE laopt::
    )

endif ()