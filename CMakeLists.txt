cmake_minimum_required(VERSION 3.9)

project(TcmSwarm CXX)

include(CTest)
include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/Vc/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NQS_USE_INTEL_MKL
"If set to ON, Intel Math Kernel Library will \
be used for BLAS and LAPACK routines." ON)

option(NQS_VERBOSE_BUILD
    "Enables some information messages during the build process." ON)

set(INTEL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/intel
    CACHE PATH "Intel root directory.")
set(MKL_ROOT ${INTEL_ROOT}/mkl
    CACHE PATH "Root directory of Intel MKL 2018 installation.")

CMAKE_DEPENDENT_OPTION(MKL_SDL
    "Use Single Dynamically Linked version of Intel MKL." OFF
    "NQS_USE_INTEL_MKL" OFF)
CMAKE_DEPENDENT_OPTION(MKL_MULTI_THREADED
    "Use multi-threaded interface of Intel MKL." ON
    "NQS_USE_INTEL_MKL" OFF)
CMAKE_DEPENDENT_OPTION(MKL_GNU_THREAD
    "Use GNU OpenMP rather than Intel OpenMP." ON
    "NQS_USE_INTEL_MKL; MKL_MULTI_THREADED" OFF)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(MKL REQUIRED)
find_package(SVML REQUIRED)
find_package(Vc REQUIRED)
find_library(gtest REQUIRED)
find_library(benchmark REQUIRED)
find_package(Backtrace REQUIRED)

message(STATUS "[Backtrace] Backtrace_LIBRARIES: ${Backtrace_LIBRARY}")
message(STATUS "[Backtrace] Backtrace_HEADER: ${Backtrace_HEADER}")
message(STATUS "[Backtrace] Backtrace_INCLUDE_DIRS: ${Backtrace_INCLUDE_DIR}")

find_package(Boost 1.65.0 REQUIRED)
add_library(boost_stacktrace INTERFACE)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_link_libraries(boost_stacktrace INTERFACE
        Boost::boost -L${Boost_LIBRARY_DIR_RELEASE}/libboost_stacktrace_basic.so
        ${CMAKE_DL_LIBS})
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_definitions(boost_stacktrace INTERFACE BOOST_STACKTRACE_USE_BACKTRACE)
    target_link_libraries(boost_stacktrace INTERFACE
        Boost::boost -L${Boost_LIBRARY_DIR_RELEASE}/libboost_stacktrace_backtrace.so
        backtrace ${CMAKE_DL_LIBS})
endif()

set(GSL_TEST OFF)
add_subdirectory(external/GSL)

set(RANGE_V3_NO_HEADER_CHECK ON)
set(RANGE_V3_NO_TESTING ON)
set(RANGE_V3_NO_EXAMPLE ON)
set(RANGE_V3_NO_PERF ON)
set(RANGES_CXX_STD 17)
set(RANGES_ASSERTIONS OFF)
set(RANGES_NATIVE ON)
add_subdirectory(external/range-v3)

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_NATIVE_SUPPORTED)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    set(NQS_WARNING_FLAGS -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-c++11-c++14-compat -Wno-weak-vtables -Wno-unused
        -Wno-exit-time-destructors -Wno-documentation-unknown-command
        -Wno-float-equal -Wno-global-constructors
        )
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    set(NQS_WARNING_FLAGS -pedantic -W -Wall -Wextra -Wcast-align -Wcast-qual
        -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
        -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
        -Wstrict-overflow=1 -Wswitch-default -Wundef -Wno-unused)
endif()

add_library(NQS INTERFACE)
target_compile_features(NQS INTERFACE cxx_std_17)
target_include_directories(NQS SYSTEM INTERFACE ${MKL_INCLUDES})
target_link_libraries(NQS INTERFACE
    GSL range-v3 Boost::boost boost_stacktrace
    ${MKL_LIBS} ${SVML_LIBS} OpenMP::OpenMP_CXX Threads::Threads)
target_compile_options(NQS INTERFACE "${WARNING_FLAGS}")

if(COMPILER_OPT_NATIVE_SUPPORTED)
    message(STATUS "[NQS] Using -march=native.")
    target_compile_options(NQS INTERFACE "-march=native")
endif()

add_executable(main main.cpp)
target_link_libraries(main PUBLIC NQS)

enable_testing()

# add_subdirectory(cbits/test)

add_library(neural SHARED cbits/nqs.h cbits/nqs.cpp)
target_compile_definitions(neural
  PRIVATE
    TCM_SWARM_NOCHECK_ALIGNMENT
    TCM_SWARM_NOCHECK_FLIPS_BOUNDS
    TCM_SWARM_NOCHECK_FLIPS_UNIQUE
    GSL_UNENFORCED_ON_CONTRACT_VIOLATION
)

target_link_libraries(neural PUBLIC NQS)


if (FALSE)

add_library(NQS INTERFACE)

target_compile_features(NQS INTERFACE cxx_std_17)
target_include_directories(NQS SYSTEM INTERFACE ${MKL_INCLUDES})
# target_include_directories(NQS INTERFACE ${Boost_INCLUDE_DIRS})
target_link_libraries(NQS
    INTERFACE
        GSL
        range-v3
        Boost::boost
        boost_stacktrace
        ${OpenMP_CXX_LIBRARIES}
        ${MKL_LIBS}
        ${SVML_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_options(NQS
    INTERFACE
        "${WARNING_FLAGS}"
        "${OpenMP_CXX_FLAGS}"
)

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_NATIVE_SUPPORTED)
if(COMPILER_OPT_NATIVE_SUPPORTED)
    target_compile_options(NQS
        INTERFACE
            "-march=native"
    )
endif()

# add_subdirectory(cbits/example)

# ==============================================================================
# 
# target_include_directories(neural
#   PUBLIC
#     ${MKL_INCLUDES}
#     ${Vc_INCLUDE_DIR}
#   )
# 
# target_compile_options(neural
#   PUBLIC
#     ${Vc_ARCHITECTURE_FLAGS}
#   )
# 
# target_compile_features(neural
#   PUBLIC
#     cxx_std_17
#   )
# 
# target_compile_definitions(neural
#   PUBLIC
#     GSL_THROW_ON_CONTRACT_VIOLATION
#     # DO_TRACE
#   )

# ==============================================================================

add_executable(main main.cpp)
target_link_libraries(main NQS)

# ==============================================================================

# add_executable(tester test.cpp)
# target_link_libraries(tester NQS)
# target_compile_definitions(tester PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)

# ==============================================================================

# add_executable(rbm-test cbits/test/rbm.cpp)
# target_compile_features(rbm-test NQS)
# target_compile_definitions(rbm-test PRIVATE GSL_THROW_ON_CONTRACT_VIOLATION)

# ==============================================================================


# add_executable(rbm-spin-test cbits/test/rbm_spin.cpp)
# 
# target_compile_features(rbm-spin-test
#   PUBLIC
#     cxx_std_17
#   )
# 
# target_compile_definitions(rbm-spin-test
#   PUBLIC
#     GSL_THROW_ON_CONTRACT_VIOLATION
#     # DO_TRACE
#   )
# 
# target_compile_options(rbm-spin-test
#   PUBLIC
#     ${Vc_ARCHITECTURE_FLAGS}
#   )
# 
# target_link_libraries(rbm-spin-test
#   PRIVATE
#     GSL
#     ${SVML_LIBS}
#     ${MKL_LIBS}
#     gtest
#     ${CMAKE_THREAD_LIBS_INIT}
#   )
# 
# target_include_directories(rbm-spin-test
#   PRIVATE
#     ${MKL_INCLUDES}
#   )

# ==============================================================================


# ==============================================================================


# ==============================================================================

add_executable(lncosh-benchmark cbits/benchmark/lncosh-benchmark.cpp)

target_compile_features(lncosh-benchmark PUBLIC cxx_std_17)

target_link_libraries(lncosh-benchmark
  PUBLIC
    GSL
    ${MKL_LIBS}
    ${SVML_LIBS}
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
  )

target_include_directories(lncosh-benchmark
  PRIVATE
    ${MKL_INCLUDES}
  )

target_compile_options(lncosh-benchmark
  PUBLIC
    ${Vc_ARCHITECTURE_FLAGS}
  )

endif()
