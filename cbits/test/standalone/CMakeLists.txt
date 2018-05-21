add_executable(standalone.simd simd.cpp)
target_link_libraries(standalone.simd NQS)
target_compile_definitions(standalone.simd PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)
add_test(test.standalone.simd standalone.simd)

add_executable(standalone.rbm_spin rbm_spin.cpp)
target_link_libraries(standalone.rbm_spin NQS)
target_compile_definitions(standalone.rbm_spin PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)
add_test(test.standalone.rbm_spin standalone.rbm_spin)

add_executable(standalone.metropolis_local metropolis_local.cpp)
target_link_libraries(standalone.metropolis_local NQS)
target_compile_definitions(standalone.metropolis_local PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)
add_test(test.standalone.metropolis_local standalone.metropolis_local)

add_executable(standalone.random random.cpp)
target_link_libraries(standalone.random NQS)
target_compile_definitions(standalone.random PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)
add_test(test.standalone.random standalone.random)

add_executable(standalone.spin spin.cpp)
target_link_libraries(standalone.spin NQS)
target_compile_definitions(standalone.spin PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)
add_test(test.standalone.spin standalone.spin)

add_executable(standalone.heisenberg heisenberg.cpp)
target_link_libraries(standalone.heisenberg NQS)
target_compile_definitions(standalone.heisenberg PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)
add_test(test.standalone.heisenberg standalone.heisenberg)

add_executable(standalone.errors errors.cpp)
target_link_libraries(standalone.errors NQS)
target_compile_definitions(standalone.errors PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)
add_test(test.standalone.errors standalone.errors)
