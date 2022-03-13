cmake_minimum_required(VERSION 3.5)

include_directories(include /usr/local/include/ ${EIGEN3_INCLUDE_DIR} /opt/homebrew/Cellar/ipopt/3.14.2/include/coin-or/ /Users/cnjones/git/lampc/src)
link_directories(/Users/cnjones/git/lampc /opt/homebrew/Cellar/ipopt/3.14.2/lib/)

add_executable(qp_make qp_make.cpp)
add_dependencies(qp_make la_compiler)
target_link_libraries(qp_make la_compiler)


add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/qp.compiled.hpp
  COMMAND qp_make
  DEPENDS qp_make
  COMMENT "Generating QP"
  VERBATIM
)

add_custom_target(
  GeneratedQP ALL
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/qp.compiled.hpp
)

# add_library(qp_lib qp_lib.cpp)
# add_dependencies(qp_lib GeneratedQP)

add_executable(qp qp.cpp)
add_dependencies(qp GeneratedQP)
# add_dependencies(qp qp_lib)
# target_link_libraries(qp qp_lib)

