cmake_minimum_required(VERSION 3.21)

project(laopt
    VERSION 0.1.0
    LANGUAGES C CXX
)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
# Keep the explicit FetchContent populate/add_subdirectory flow for CMake 3.21
# support while still excluding third-party install rules.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
    cmake_policy(SET CMP0169 OLD)
endif()

include(GNUInstallDirs)

#### Local build options ####
option(LAOPT_WITH_CASADI "Build local tests/examples with CasADi support" OFF)
option(LAOPT_WITH_IPOPT "Build local tests/examples with IPOPT support" OFF)
option(LAOPT_WITH_OSQP "Build local tests/examples with OSQP support" OFF)
option(LAOPT_WITH_QPALM "Build local tests/examples with QPALM support" OFF)
option(LAOPT_WITH_PROXQP "Build local tests/examples with ProxQP support" OFF)
option(LAOPT_WITH_QPSWIFT "Build local tests/examples with qpSWIFT support" OFF)
option(LAOPT_WITH_PIQP "Build local tests/examples with PIQP support" OFF)
option(LAOPT_WITH_HPIPM "Build local tests/examples with HPIPM support" OFF)

#### Tests/Benchmarks options ####
option(LAOPT_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(LAOPT_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(LAOPT_BUILD_BENCHMARKS "Build benchmarks" OFF)

#### Developer options ####
option(LAOPT_ENABLE_SANITIZERS "Build local targets with sanitizers enabled" OFF)

#### Install options ####
option(LAOPT_ENABLE_INSTALL "Enable install targets" ${PROJECT_IS_TOP_LEVEL})

set(LAOPT_BUILD_TESTS_ENABLED OFF)
if (LAOPT_BUILD_TESTS)
    include(CTest)
    if (BUILD_TESTING)
        set(LAOPT_BUILD_TESTS_ENABLED ON)
    endif()
endif()

# Set build type to Release by default for single-config generators.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if (CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Configuration types: ${CMAKE_CONFIGURATION_TYPES}")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# Compiler and linker flags for local tests/examples/benchmarks.
set(gcc_like_cxx CXX ARMClang AppleClang Clang GNU LCC)
set(msvc_like CXX MSVC)
set(laopt_compile_options)
set(laopt_link_options)
if (CMAKE_CXX_COMPILER_ID IN_LIST gcc_like_cxx)
    list(APPEND laopt_compile_options -Wall -Wextra -Wno-unused-parameter -ftemplate-backtrace-limit=0)
elseif (CMAKE_CXX_COMPILER_ID IN_LIST msvc_like)
    list(APPEND laopt_compile_options /W4)
endif()
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

if (LAOPT_ENABLE_SANITIZERS)
    if (CMAKE_CXX_COMPILER_ID IN_LIST gcc_like_cxx)
        list(APPEND laopt_compile_options -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
        list(APPEND laopt_link_options -fsanitize=address -fsanitize=undefined)
    elseif (CMAKE_CXX_COMPILER_ID IN_LIST msvc_like)
        list(APPEND laopt_compile_options /fsanitize=address)
    endif()
    message(STATUS "Building local targets with sanitizer compile flags: ${laopt_compile_options}")
    if (laopt_link_options)
        message(STATUS "Building local targets with sanitizer link flags: ${laopt_link_options}")
    endif()
endif()

# Find Eigen3.4+.
if (DEFINED EIGEN3_INCLUDE_DIRS AND NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN3_INCLUDE_DIRS}")
else()
    find_package(Eigen3 REQUIRED NO_MODULE)

    if (Eigen3_VERSION VERSION_LESS "3.4")
        message(FATAL_ERROR "Eigen3 >= 3.4 required, found ${Eigen3_VERSION}")
    endif()
endif()
get_target_property(LAOPT_EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Using Eigen3 from: ${LAOPT_EIGEN_INCLUDE_DIRS}")

add_library(laopt INTERFACE)
target_include_directories(laopt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(laopt INTERFACE Eigen3::Eigen)
target_compile_definitions(laopt INTERFACE EIGEN_STACK_ALLOCATION_LIMIT=1000000)
target_compile_features(laopt INTERFACE cxx_std_17)

add_library(laopt::laopt ALIAS laopt)

function(laopt_get_dependency_scope target_name out_var)
    get_target_property(target_type ${target_name} TYPE)
    if (target_type STREQUAL "INTERFACE_LIBRARY")
        set(${out_var} INTERFACE PARENT_SCOPE)
    else()
        set(${out_var} PRIVATE PARENT_SCOPE)
    endif()
endfunction()

function(laopt_target_enable_casadi target_name)
    if (NOT LAOPT_WITH_CASADI)
        return()
    endif()
    laopt_get_dependency_scope(${target_name} dependency_scope)
    find_package(Casadi REQUIRED)
    target_compile_definitions(${target_name} ${dependency_scope} LAOPT_WITH_CASADI)
    target_link_libraries(${target_name} ${dependency_scope} casadi)
endfunction()

function(laopt_target_enable_ipopt target_name)
    if (NOT LAOPT_WITH_IPOPT)
        return()
    endif()
    laopt_get_dependency_scope(${target_name} dependency_scope)
    find_package(Ipopt REQUIRED)
    target_compile_definitions(${target_name} ${dependency_scope} LAOPT_WITH_IPOPT)
    target_link_libraries(${target_name} ${dependency_scope} Ipopt::Ipopt)
endfunction()

function(laopt_target_enable_osqp target_name)
    if (NOT LAOPT_WITH_OSQP)
        return()
    endif()
    laopt_get_dependency_scope(${target_name} dependency_scope)
    find_package(osqp REQUIRED)
    target_compile_definitions(${target_name} ${dependency_scope} LAOPT_WITH_OSQP)
    target_link_libraries(${target_name} ${dependency_scope} osqp::osqp)
endfunction()

function(laopt_target_enable_qpalm target_name)
    if (NOT LAOPT_WITH_QPALM)
        return()
    endif()
    laopt_get_dependency_scope(${target_name} dependency_scope)
    find_package(QPALM REQUIRED)
    target_compile_definitions(${target_name} ${dependency_scope} LAOPT_WITH_QPALM)
    target_link_libraries(${target_name} ${dependency_scope} QPALM::qpalm)
endfunction()

function(laopt_target_enable_proxqp target_name)
    if (NOT LAOPT_WITH_PROXQP)
        return()
    endif()
    laopt_get_dependency_scope(${target_name} dependency_scope)
    find_package(proxsuite REQUIRED)
    target_compile_definitions(${target_name} ${dependency_scope} LAOPT_WITH_PROXQP)
    target_link_libraries(${target_name} ${dependency_scope} proxsuite::proxsuite)
endfunction()

function(laopt_target_enable_qpswift target_name)
    if (NOT LAOPT_WITH_QPSWIFT)
        return()
    endif()
    laopt_get_dependency_scope(${target_name} dependency_scope)
    find_package(qpSWIFT REQUIRED)
    target_compile_definitions(${target_name} ${dependency_scope} LAOPT_WITH_QPSWIFT)
    target_link_libraries(${target_name} ${dependency_scope} qpSWIFT::qpSWIFT-static)
endfunction()

function(laopt_target_enable_piqp target_name)
    if (NOT LAOPT_WITH_PIQP)
        return()
    endif()
    laopt_get_dependency_scope(${target_name} dependency_scope)
    find_package(piqp REQUIRED)
    target_compile_definitions(${target_name} ${dependency_scope} LAOPT_WITH_PIQP)
    target_link_libraries(${target_name} ${dependency_scope} piqp::piqp_header_only)
endfunction()

function(laopt_target_enable_hpipm target_name)
    if (NOT LAOPT_WITH_HPIPM)
        return()
    endif()
    laopt_get_dependency_scope(${target_name} dependency_scope)
    find_package(blasfeo REQUIRED)
    find_package(hpipm REQUIRED)
    target_compile_definitions(${target_name} ${dependency_scope} LAOPT_WITH_HPIPM)
    target_link_libraries(${target_name} ${dependency_scope} blasfeo hpipm)
endfunction()

if (LAOPT_BUILD_TESTS_ENABLED)
    add_subdirectory(tests)
endif()
if (LAOPT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if (LAOPT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if (LAOPT_ENABLE_INSTALL)
    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
        TARGETS laopt
        EXPORT laoptTargets
    )

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
endif()
