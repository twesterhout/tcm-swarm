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

// This file implements C++ wrappers around MKL's cblas_?gemv functions.

#ifndef TCM_SWARM_DETAIL_GEMV_HPP
#define TCM_SWARM_DETAIL_GEMV_HPP

#include <gsl/span>
#include <gsl/multi_span>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace {

// clang-format off
#define TCM_SWARM_GEMV_SIGNATURE_FOR(T)                                     \
    auto gemv(CBLAS_LAYOUT const layout, CBLAS_TRANSPOSE const trans_A,     \
        difference_type const m, difference_type const n,                   \
        T const alpha, T const* const a, difference_type const ld_a,        \
        T const* const x, difference_type const inc_x,                      \
        T const beta, T* y, difference_type const inc_y) noexcept -> void
        // clang-format on

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_GEMV_SIGNATURE_FOR(float)
        {
            cblas_sgemv(layout, trans_A, m, n, alpha, a, ld_a, x,
                inc_x, beta, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_GEMV_SIGNATURE_FOR(double)
        {
            cblas_dgemv(layout, trans_A, m, n, alpha, a, ld_a, x,
                inc_x, beta, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_GEMV_SIGNATURE_FOR(std::complex<float>)
        {
            cblas_cgemv(layout, trans_A, m, n, &alpha, a, ld_a, x,
                inc_x, &beta, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_GEMV_SIGNATURE_FOR(std::complex<double>)
        {
            cblas_zgemv(layout, trans_A, m, n, &alpha, a, ld_a, x,
                inc_x, &beta, y, inc_y);
        }
    } // namespace
} // namespace detail

/// Let \f$V, W\f$ be vector spaces over scalar field \f$\mathbb{F}\f$. Then
/// #gemv: \f$\mathbb{F} \to \operatorname{Lin}(V, W) \to V \to \mathbb{F} \to
/// W \to \{\ast\} \f$ is defined by: \f$(\alpha, A, x, \beta, y) \mapsto
/// y := \alpha A x + \beta y\f$.
struct gemv_fn {
    /// \brief Implementation for #gsl::span.
    ///
    /// It might seem surprising at first that the matrix is passed as a span
    /// (i.e. "pointer + length") combination rather than the standard "pointer
    /// + ldim + height + width" combination. However, if we restrict ourselves
    /// to cases when leading dimension is equal to either height or width
    /// (depending on the layout) these representations become equivalent.
    ///
    /// Suppose that `trans == Transpose::None`, then we have more or less the
    /// following:
    /// \code{.unparsed}
    ///       n
    ///    ←-----→
    ///  ↑ ⌈     ⌉ ⌈ ⌉    ⌈ ⌉    ↑  ↑
    ///  | |     | |x|    | |    |n |
    /// m| |  a  | ⌊ ⌋  + |y|    ↓  |m
    ///  | |     |        | |       |
    ///  ↓ ⌊     ⌋        ⌊ ⌋       ↓
    /// \endcode
    /// i.e. `n = x.size()` and `m = x.size()`. If `trans == Transpose::ConjTrans`
    /// or `trans == Transpose::Trans`, we get
    /// \code{.unparsed}
    ///           m
    ///    ←-------------→
    ///  ↑ ⌈       T     ⌉ ⌈ ⌉    ⌈ ⌉    ↑  ↑
    /// n| |      a      | | |    |y|    |n |
    ///  ↓ ⌊             ⌋ |x|  + ⌊ ⌋    ↓  |m
    ///                    | |              |
    ///                    ⌊ ⌋              ↓
    /// \endcode
    /// and thus `n = y.size()` and `m = x.size()`.
    ///
    /// \see [cblas_?gemv](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemv)
    template <class T, std::ptrdiff_t DimA, std::ptrdiff_t DimX, std::ptrdiff_t DimY>
    TCM_SWARM_FORCEINLINE auto operator()(Layout const layout,
        Transpose const trans, T const alpha, gsl::span<T const, DimA> const a,
        gsl::span<T const, DimX> const x, T beta, gsl::span<T, DimY> const y) const
        noexcept(!::tcm::detail::gsl_can_throw()) -> void
    {
        Expects(a.size() / x.size() == y.size());
        // clang-format off
        auto const [m, n] = (trans == Transpose::None)
            ? std::make_tuple(
                  gsl::narrow<difference_type>(y.size()),
                  gsl::narrow<difference_type>(x.size()))
            : std::make_tuple(
                  gsl::narrow<difference_type>(x.size()),
                  gsl::narrow<difference_type>(y.size()));
        // clang-format on
        auto const ldim = (layout == Layout::ColMajor) ? m : n;
        constexpr difference_type one{1};
        detail::gemv(to_raw_enum(layout), to_raw_enum(trans), m, n, alpha,
            a.data(), ldim, x.data(), one, beta, y.data(), one);
    }

    // clang-format off
    template <class T>
    TCM_SWARM_FORCEINLINE
    auto operator()(Layout const layout, Transpose const trans,
        difference_type const m, difference_type const n,
        T const alpha, T const* const a, difference_type const ld_a,
        T const* const x, difference_type const inc_x,
        T beta, T* const y, difference_type const inc_y)
        // clang-format on
        const noexcept -> void
    {
        detail::gemv(to_raw_enum(layout), to_raw_enum(trans), m, n, alpha, a,
            ld_a, x, inc_x, beta, y, inc_y);
    }

    // clang-format off
    template <class T, std::ptrdiff_t DimA1, std::ptrdiff_t DimA2,
        std::ptrdiff_t DimX, std::ptrdiff_t DimY>
    TCM_SWARM_FORCEINLINE
    auto operator()(Layout const layout, Transpose const trans,
        T const alpha, gsl::multi_span<T const, DimA1, DimA2> const a,
        gsl::span<T const, DimX> const x,
        T beta, gsl::span<T, DimY> const y) const
        noexcept(!::tcm::detail::gsl_can_throw()) -> void
    // clang-format on
    {
        // clang-format off
        auto const [m, n] = (layout == Layout::RowMajor)
            ? std::make_tuple(
                  gsl::narrow<difference_type>(a.template extent<0>()),
                  gsl::narrow<difference_type>(a.template extent<1>()))
            : std::make_tuple(
                  gsl::narrow<difference_type>(a.template extent<1>()),
                  gsl::narrow<difference_type>(a.template extent<0>()));
        // clang-format on
        auto const ldim =
            gsl::narrow_cast<difference_type>(a.template extent<1>());
        constexpr difference_type one{1};

        if (trans == Transpose::None) {
            Expects(gsl::narrow<difference_type>(y.size()) == m);
            Expects(gsl::narrow<difference_type>(x.size()) == n);
        }
        else {
            Expects(gsl::narrow<difference_type>(x.size()) == m);
            Expects(gsl::narrow<difference_type>(y.size()) == n);
        }

        detail::gemv(to_raw_enum(layout), to_raw_enum(trans), m, n, alpha,
            a.data(), ldim, x.data(), one, beta, y.data(), one);
    }
};

/// \brief Global instance of #tcm::mkl::gemv_fn.
TCM_SWARM_INLINE_VARIABLE(gemv_fn, gemv)

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_GEMV_HPP

