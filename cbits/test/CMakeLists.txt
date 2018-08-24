
add_library(nqs_test SHARED)
target_sources(nqs_test PRIVATE parser.cpp)
target_link_libraries(nqs_test PUBLIC nqs_cxx gtest stdc++fs)

add_executable(rbm_spin rbm_spin.cpp)
target_link_libraries(rbm_spin PRIVATE nqs_test)
add_test(NAME test.rbm_spin COMMAND rbm_spin WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(metropolis_local metropolis_local.cpp)
target_link_libraries(metropolis_local PRIVATE nqs_test)
add_test(NAME test.metropolis_local COMMAND metropolis_local WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(force force.cpp)
target_link_libraries(force PRIVATE nqs_test)
add_test(NAME test.force COMMAND force WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# 
# add_executable(axpy axpy.cpp)
# target_link_libraries(axpy NQS gtest stdc++fs)
# target_compile_options(axpy PRIVATE -m64)
# target_compile_definitions(axpy PRIVATE MKL_ILP64 GSL_THROW_ON_CONTRACT_VIOLATION)
# add_test(NAME test.axpy COMMAND axpy WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# 
# add_executable(lncosh lncosh.cpp)
# target_compile_features(lncosh PRIVATE cxx_std_17)
# target_link_libraries(lncosh NQS gtest)
# target_compile_definitions(lncosh PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)
# add_test(NAME test.log_cosh COMMAND lncosh WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# 
# add_subdirectory(standalone)
