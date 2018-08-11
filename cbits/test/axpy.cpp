// Copyright Tom Westerhout (c) 2018
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Tom Westerhout nor the names of other
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <complex>

#include <gtest/gtest.h>

// #include "../detail/axpy.hpp"
// #include "../rbm_spin.hpp"
#include <mkl_cblas.h>

#define EXPECT_CFLOAT_NEAR(val1, val2, abs_error)                         \
    EXPECT_NEAR(val1.real(), val2.real(), abs_error);                     \
    EXPECT_NEAR(val1.imag(), val2.imag(), abs_error);

#if 0
extern "C" {
void cblas_caxpy(int, void const*, void const*, int,
    void*, int);
} // external "C"
#endif

TEST(Axpy, CFloat)
{
    // using Rbm = tcm::RbmBase<std::complex<float>>;
    using vector_type = std::vector<std::complex<float>>; // Rbm::vector_type;

    vector_type x(5000);
    vector_type y(5000);
    std::complex<float> x0 = {4.6f, 0.01f};
    std::complex<float> y0 = {-5.0f, 2.3f};
    std::complex<float> a = {1.0f, 0};

    std::fill(std::begin(x), std::end(x), x0);
    std::fill(std::begin(y), std::end(y), y0);

#if 0
    tcm::mkl::axpy(a,
        gsl::span<std::complex<float> const>{x},
        gsl::span<std::complex<float>>{y});
#elif 1
    cblas_caxpy(5000, &a, x.data(), 1, y.data(), 1);
#else
    int n = 5000;
    int inc = 1;
    caxpy_(&n, &a, x.data(), &inc, y.data(), &inc);
#endif

    for (unsigned i = 0; i < x.size(); ++i) {
        EXPECT_CFLOAT_NEAR(y[i], (a * x0 + y0), 1.0E-5f);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
