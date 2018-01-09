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

#ifndef TCM_SWARM_DETAIL_LNCOSH_HPP
#define TCM_SWARM_DETAIL_LNCOSH_HPP

#include <cmath>
#include <complex>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"


TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

struct lncosh_fn {
  private:

#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#endif
    template <class T>
    static inline constexpr T ln2{
        0.69314718055994530941723212145817656807550};
#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic push
#endif

  public:
    auto operator()(float const x) const noexcept -> float
    {
        constexpr auto const threshold = 10.0f;
        if (std::abs(x) > threshold) { return x - ln2<float>; }
        return std::log(std::cosh(x));
    }

    auto operator()(double const x) const noexcept -> double
    {
        constexpr auto const threshold = 15.0;
        if (std::abs(x) > threshold) { return x - ln2<double>; }
        return std::log(std::cosh(x));
    }

    template <class T>
    auto operator()(std::complex<T> const z) const noexcept
        -> std::complex<T>
    {
        auto const x = z.real();
        auto const y = z.imag();
        return this->operator()(x)
               + std::log(std::complex<T>{
                     std::cos(y), std::tanh(x) * std::sin(y)});
    }
};

/// \brief Computes unconjugated product of two vectors.
TCM_SWARM_INLINE_VARIABLE(lncosh_fn, lncosh);

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_LNCOSH_HPP
