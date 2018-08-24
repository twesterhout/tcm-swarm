# - Find Intel SVML
#

include(FindPackageHandleStandardArgs)

if(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    set(MKL_ARCH "64")
    set(MKL_ARCH_DIR "intel64")
else()
    set(MKL_ARCH "32")
    set(MKL_ARCH_DIR "ia32")
endif()

find_library(INTEL_SVML_LIB svml
    PATHS ${INTEL_ROOT}/lib/${MKL_ARCH_DIR}/)

find_library(INTEL_INTLC_LIB intlc
    PATHS ${INTEL_ROOT}/lib/${MKL_ARCH_DIR}/)
# PATHS ${INTEL_ROOT}/lib/${MKL_ARCH_DIR}/)

find_package_handle_standard_args(SVML DEFAULT_MSG
    INTEL_SVML_LIB INTEL_INTLC_LIB)

if(SVML_FOUND)
    set(SVML_LIBS ${INTEL_SVML_LIB} ${INTEL_INTLC_LIB})
endif()

