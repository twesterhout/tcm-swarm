

add_executable(rbm-spin rbm_spin.cpp)
target_link_libraries(rbm-spin NQS)
target_compile_definitions(rbm-spin GSL_THROW_ON_CONTRACT_VIOLATION)

add_executable(lncosh lncosh.cpp)
target_compile_features(lncosh cxx_std_17)
target_link_libraries(lncosh GSL ${SVML_LIBS} gtest  ${CMAKE_THREAD_LIBS_INIT})
target_compile_definitions(lncosh PUBLIC GSL_THROW_ON_CONTRACT_VIOLATION)
if(COMPILER_OPT_NATIVE_SUPPORTED)
    target_compile_options(lncosh "-march=native")
endif()
