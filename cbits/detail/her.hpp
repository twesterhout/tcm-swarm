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

// This file implements C++ wrappers around cblas_?her functions.

#ifndef TCM_SWARM_DETAIL_HER_HPP
#define TCM_SWARM_DETAIL_HER_HPP

#include <gsl/span>
#include <gsl/multi_span>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace {

#define TCM_SWARM_HER_SIGNATURE_FOR(R, C)                                 \
    auto her(CBLAS_LAYOUT const layout, CBLAS_UPLO const uplo,            \
        difference_type const n, R const alpha, C const* const x,         \
        difference_type const inc_x, C* const a,                          \
        difference_type const ld_a) noexcept->void

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_HER_SIGNATURE_FOR(float, std::complex<float>)
        {
            cblas_cher(layout, uplo, n, alpha, x, inc_x, a, ld_a);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_HER_SIGNATURE_FOR(double, std::complex<double>)
        {
            cblas_zher(layout, uplo, n, alpha, x, inc_x, a, ld_a);
        }
#undef TCM_SWARM_HER_SIGNATURE_FOR

    } // namespace
} // namespace detail

struct her_fn {
    // clang-format off
    template <class T, std::ptrdiff_t Dim>
    TCM_SWARM_FORCEINLINE
    auto operator()(Layout const layout, UpLo const uplo,
        T const alpha, gsl::span<std::complex<T> const, Dim> const x,
        gsl::multi_span<std::complex<T>, Dim, Dim> const a) const
            noexcept(!::tcm::detail::gsl_can_throw()) -> void
    {
        Expects(a.template extent<0>() == a.template extent<1>());
        Expects(a.template extent<0>() == x.size());
        auto const n = gsl::narrow<difference_type>(x.size());
        auto const ld_a = gsl::narrow<difference_type>(a.template extent<1>());
        constexpr difference_type one{1};

        detail::her(to_raw_enum(layout), to_raw_enum(uplo),
            n, alpha, x.data(), one, a.data(), ld_a);
    }
    // clang-format on
};

/// \brief Global instance of #tcm::mkl::her_fn.
TCM_SWARM_INLINE_VARIABLE(her_fn, her)

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_HER_HPP

