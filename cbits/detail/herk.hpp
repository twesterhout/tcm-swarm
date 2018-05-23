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

// This file implements C++ wrappers around cblas_?herk functions.

#ifndef TCM_SWARM_DETAIL_HERK_HPP
#define TCM_SWARM_DETAIL_HERK_HPP

#include <gsl/span>
#include <gsl/multi_span>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace {

#define TCM_SWARM_HERK_SIGNATURE_FOR(R, C)                                \
    auto herk(CBLAS_LAYOUT const layout, CBLAS_UPLO const uplo,           \
        CBLAS_TRANSPOSE const trans_A, difference_type const n,           \
        difference_type const k, R const alpha, C const* const a,         \
        difference_type const ld_a, R const beta, C* const c,             \
        difference_type const ld_c) noexcept->void

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_HERK_SIGNATURE_FOR(float, std::complex<float>)
        {
            cblas_cherk(layout, uplo, trans_A, n, k, alpha, a, ld_a, beta,
                c, ld_c);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_HERK_SIGNATURE_FOR(double, std::complex<double>)
        {
            cblas_zherk(layout, uplo, trans_A, n, k, alpha, a, ld_a, beta,
                c, ld_c);
        }

    } // namespace
} // namespace detail

struct herk_fn {
    // clang-format off
    template <class T, std::ptrdiff_t DimA1, std::ptrdiff_t DimA2,
        std::ptrdiff_t DimC>
    TCM_SWARM_FORCEINLINE
    auto operator()(Layout const layout, UpLo const uplo, Transpose const trans,
        T const alpha, gsl::multi_span<std::complex<T> const, DimA1, DimA2> const a,
        T const beta, gsl::multi_span<std::complex<T>, DimC, DimC> const c) const
            noexcept(!::tcm::detail::gsl_can_throw()) -> void
    {
        Expects(c.template extent<0>() == c.template extent<1>());
        auto const n = gsl::narrow<difference_type>(c.template extent<0>());
        auto const k = ((trans == Transpose::None && layout == Layout::RowMajor)
                || (trans == Transpose::ConjTrans && layout == Layout::ColMajor))
            ? gsl::narrow<difference_type>(a.template extent<1>())
            : gsl::narrow<difference_type>(a.template extent<0>());
        auto const ld_a = gsl::narrow<difference_type>(a.template extent<1>());
        auto const ld_c = gsl::narrow<difference_type>(c.template extent<1>());

        if ((trans == Transpose::None && layout == Layout::RowMajor)
            || (trans == Transpose::ConjTrans && layout == Layout::ColMajor)) {
            Expects(
                gsl::narrow<difference_type>(a.template extent<0>()) == n);
        }
        else {
            Expects(
                gsl::narrow<difference_type>(a.template extent<1>()) == n);
        }

        detail::herk(to_raw_enum(layout), to_raw_enum(uplo),
            to_raw_enum(trans), n, k, alpha, a.data(), ld_a, beta,
            c.data(), ld_c);
    }
    // clang-format on
};

/// \brief Global instance of #tcm::mkl::herk_fn.
TCM_SWARM_INLINE_VARIABLE(herk_fn, herk)

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_HERK_HPP

