find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    pkg_check_modules(PC_Ipopt QUIET ipopt)
endif()

set(Ipopt_DEFINITIONS ${PC_Ipopt_CFLAGS_OTHER})

find_path(Ipopt_INCLUDE_DIRS
    NAMES IpIpoptNLP.hpp
    HINTS
        ${PC_Ipopt_INCLUDEDIR}
        ${PC_Ipopt_INCLUDE_DIRS}
    PATHS
        "${CMAKE_INSTALL_PREFIX}/include"
)

find_library(Ipopt_LIBRARIES
    NAMES ipopt
    HINTS
        ${PC_Ipopt_LIBDIR}
        ${PC_Ipopt_LIBRARY_DIRS}
)

set(Ipopt_VERSION ${PC_Ipopt_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ipopt
    REQUIRED_VARS Ipopt_INCLUDE_DIRS Ipopt_LIBRARIES
    VERSION_VAR Ipopt_VERSION
)

if (Ipopt_FOUND AND NOT TARGET Ipopt::Ipopt)
    add_library(Ipopt::Ipopt UNKNOWN IMPORTED)
    set_target_properties(Ipopt::Ipopt PROPERTIES
        IMPORTED_LOCATION "${Ipopt_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${Ipopt_INCLUDE_DIRS}"
    )
    if (Ipopt_DEFINITIONS)
        set_property(TARGET Ipopt::Ipopt PROPERTY INTERFACE_COMPILE_OPTIONS "${Ipopt_DEFINITIONS}")
    endif()
    if (PC_Ipopt_LDFLAGS_OTHER)
        set_property(TARGET Ipopt::Ipopt PROPERTY INTERFACE_LINK_OPTIONS "${PC_Ipopt_LDFLAGS_OTHER}")
    endif()
endif()

mark_as_advanced(Ipopt_INCLUDE_DIRS Ipopt_LIBRARIES)
