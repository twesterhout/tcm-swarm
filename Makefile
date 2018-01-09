CC  = clang
CXX = clang++

CLANG_WARNINGS := \
 -W -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic \
 -fcomment-block-commands=cond

GCC_WARNINGS := \
 -W -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy \
 -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op \
 -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow \
 -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wvariadic-macros \
 -Wparentheses -Wnoexcept -Wnoexcept-type \

_MKL_ROOT   := /opt/intel/compilers_and_libraries_2018.0.128/linux/mkl
MKL_INCLUDE := -isystem${_MKL_ROOT}/include
MKL_LIB     := -L${_MKL_ROOT}/lib/intel64 \
 -Wl,-rpath=${_MKL_ROOT}/lib/intel64 -lmkl_rt

CXX_FLAGS = -std=c++17 -O3 -g -ggdb -fvectorize -msse2 -mavx \
  $(CLANG_WARNINGS) $(MKL_INCLUDE)

libneural.so: cbits/neural.cpp cbits/neural.h cbits/RBM.hpp cbits/detail/*.hpp
	$(CXX) $(CXX_FLAGS) -fPIC -shared -DNO_DO_TRACE \
		-Icbits cbits/neural.cpp \
		-o libneural.so


.PHONY: clean check doc

clean:
	rm -f libneural.so

check: cbits/neural.cpp cbits/neural.h cbits/RBM.hpp cbits/detail/*.hpp
	clang-tidy \
		-extra-arg="-std=c++17" \
		-extra-arg="$(MKL_INCLUDE)" \
		-extra-arg="$(MKL_ALLOCATOR_INCLUDE)" \
		cbits/RBM.hpp \
		cbits/detail/*.hpp \
		cbits/neural.cpp
	clang-tidy \
		-extra-arg="-std=c89" \
		-extra-arg="$(MKL_INCLUDE)" \
		cbits/neural.h

doc: doc/source/*.rst
	cd doc; \
	make singlehtml; \
	cd ..
