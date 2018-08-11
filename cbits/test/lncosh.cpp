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

#include <cmath>
#include <random>

#include <Vc/Vc>
#include <gtest/gtest.h>

#include "../detail/mkl_allocator.hpp"
#include "../detail/simd.hpp"

#define EXPECT_CFLOAT_NEAR(val1, val2, abs_error)                         \
    EXPECT_NEAR(val1.real(), val2.real(), abs_error);                     \
    EXPECT_NEAR(val1.imag(), val2.imag(), abs_error);

#define ASSERT_CFLOAT_NEAR(val1, val2, abs_error)                         \
    ASSERT_NEAR(val1.real(), val2.real(), abs_error);                     \
    ASSERT_NEAR(val1.imag(), val2.imag(), abs_error);

template <class T, class Abi>
auto log_cosh_calculation(T const min, T const max, T const eps)
{
    using V   = Vc::simd<T, Abi>;

    alignas(Vc::memory_alignment_v<V>) T in_real[V::size()];
    alignas(Vc::memory_alignment_v<V>) T in_imag[V::size()];
    alignas(Vc::memory_alignment_v<V>) T out_real[V::size()];
    alignas(Vc::memory_alignment_v<V>) T out_imag[V::size()];
    std::mt19937                      gen{1243};
    std::uniform_real_distribution<T> dist{min, max};
    auto const random = [&dist, &gen]() { return dist(gen); };

    for (auto iterations = 10000; iterations != 0; --iterations) {
        std::generate(std::begin(in_real), std::end(in_real), random);
        std::generate(std::begin(in_imag), std::end(in_imag), random);

        auto [lncosh_x, lncosh_y] =
            tcm::_log_cosh<T, Abi>(V{in_real, Vc::flags::vector_aligned},
                V{in_imag, Vc::flags::vector_aligned});
        lncosh_x.copy_to(out_real, Vc::flags::vector_aligned);
        lncosh_y.copy_to(out_imag, Vc::flags::vector_aligned);

        for (std::size_t i = 0; i < V::size(); ++i) {
            auto lncosh = std::log(
                std::cosh(std::complex<long double>{in_real[i], in_imag[i]}));
            ASSERT_NEAR(out_real[i], lncosh.real(),
                std::max(1.0E-5l, eps * std::abs(lncosh.real())));
            ASSERT_NEAR(
                out_imag[i], lncosh.imag(), eps * std::abs(lncosh.imag()));
        }
    }
}

#if 0
TEST(LnCosh, FloatAVX)
{
    using T = float;
    using Abi = Vc::simd_abi::avx;
    using V = Vc::simd<T, Abi>;

    constexpr T eps{5.0E-3};
    alignas(Vc::memory_alignment_v<V>) T in_real[V::size()];
    alignas(Vc::memory_alignment_v<V>) T in_imag[V::size()];
    alignas(Vc::memory_alignment_v<V>) T out_real[V::size()];
    alignas(Vc::memory_alignment_v<V>) T out_imag[V::size()];
    std::mt19937                      gen{1243};
    constexpr T                       min{-30};
    constexpr T                       max{30};
    std::uniform_real_distribution<T> dist{min, max};
    auto const random = [&dist, &gen]() { return dist(gen); };

    for (auto iterations = 10000; iterations != 0; --iterations) {
        std::generate(std::begin(in_real), std::end(in_real), random);
        std::generate(std::begin(in_imag), std::end(in_imag), random);

        // std::cout << "-> " << V{in_real, Vc::flags::vector_aligned} << '\n';
        // std::cout << "-> " << V{in_imag, Vc::flags::vector_aligned} << '\n';
        auto [lncosh_x, lncosh_y] =
            tcm::_log_cosh<T, Abi>(V{in_real, Vc::flags::vector_aligned},
                V{in_imag, Vc::flags::vector_aligned});
        // std::cout << lncosh_x << '\n';
        // std::cout << lncosh_y << '\n';
        lncosh_x.copy_to(out_real, Vc::flags::vector_aligned);
        lncosh_y.copy_to(out_imag, Vc::flags::vector_aligned);

        for (std::size_t i = 0; i < V::size(); ++i) {
            auto lncosh =
                std::log(std::cosh(std::complex<long double>{in_real[i], in_imag[i]}));
            // std::cout << lncosh << '\n';
            ASSERT_NEAR(
                out_imag[i], lncosh.imag(), eps * std::abs(lncosh.imag()));
            ASSERT_NEAR(
                out_real[i], lncosh.real(), eps * std::abs(lncosh.real()));
        }
    }
}
#endif


template <class T>
TCM_SWARM_NOINLINE auto lncosh_stl(gsl::span<T const> const x) noexcept -> T
{
    return std::accumulate(std::begin(x), std::end(x), T{0},
        [](auto const acc, auto const a) {
            auto b = std::complex<long double>{a.real(), a.imag()};
            b = std::log(std::cosh(b));
            using R = typename T::value_type;
            return acc
                   + T{static_cast<R>(b.real()), static_cast<R>(b.imag())};
        });
}

template <class T, class Abi>
auto sum_log_cosh_calculation(int const size)
{
    std::vector<std::complex<T>,
        tcm::mkl::mkl_allocator<std::complex<T>, 64>>
                                      x(static_cast<unsigned>(size));
    std::mt19937                      gen{1253};
    constexpr T                       min{-30};
    constexpr T                       max{30};
    std::uniform_real_distribution<T> dist{min, max};
    auto const random = [&dist, &gen]() { return dist(gen); };

    for (auto iterations = 1000; iterations != 0; --iterations) {
        std::generate(std::begin(x), std::end(x), random);

        // std::cout << "-> " << V{in_real, Vc::flags::vector_aligned} << '\n';
        // std::cout << "-> " << V{in_imag, Vc::flags::vector_aligned} << '\n';
        auto const sum_log_cosh_x =
            tcm::sum_log_cosh<T, Abi>(gsl::span<std::complex<T> const>{x});

        auto const sum_log_cosh_x_expected =
            lncosh_stl(gsl::span<std::complex<T> const>{x});

        ASSERT_CFLOAT_NEAR(sum_log_cosh_x, sum_log_cosh_x_expected,
            T{1.0E-3} * std::abs(sum_log_cosh_x_expected));
    }
}

#define SUM_LOG_COSH(n)                                                   \
    TEST(SumLnCosh, Float##n)                                             \
    {                                                                     \
        sum_log_cosh_calculation<float, Vc::simd_abi::native<float>>(n);  \
    }

TEST(LogCosh, FloatSSE)
{
    log_cosh_calculation<float, Vc::simd_abi::sse>(-30, 30, 1.0E-4);
}

TEST(LogCosh, FloatAVX)
{
    log_cosh_calculation<float, Vc::simd_abi::avx>(-30, 30, 1.0E-4);
}

SUM_LOG_COSH(20)
SUM_LOG_COSH(23)
SUM_LOG_COSH(59)

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

