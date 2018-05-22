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

// This file implements a C++ wrapper around MKL's cblas_?axpy functions.

#ifndef TCM_SWARM_DETAIL_AXPY_HPP
#define TCM_SWARM_DETAIL_AXPY_HPP

#include <gsl/span>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace /*anonymous*/ {

#define TCM_SWARM_AXPY_SIGNATURE_FOR(C, T)                                \
    auto axpy(difference_type n, C const alpha, T const* const x,         \
        difference_type const inc_x, T* const y,                          \
        difference_type const inc_y) noexcept->void

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_AXPY_SIGNATURE_FOR(float, float)
        {
            cblas_saxpy(n, alpha, x, inc_x, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_AXPY_SIGNATURE_FOR(double, double)
        {
            cblas_daxpy(n, alpha, x, inc_x, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_AXPY_SIGNATURE_FOR(
            std::complex<float>, std::complex<float>)
        {
            cblas_caxpy(n, &alpha, x, inc_x, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_AXPY_SIGNATURE_FOR(
            std::complex<double>, std::complex<double>)
        {
            cblas_zaxpy(n, &alpha, x, inc_x, y, inc_y);
        }

#undef TCM_SWARM_AXPY_SIGNATURE_FOR
    } // namespace
} // namespace detail

/// Let \f$V\f$ be a vector space over scalar field \f$\mathbb{F}\f$. Then
/// #axpy: \f$\mathbb{F} \to V \to V \to \{\ast\}\f$ is
/// defined by: \f$(\alpha, x, y) \mapsto y := \alpha x + y\f$.
struct axpy_fn {
    /// \brief Implementation for #gsl::span.
    ///
    /// \see [cblas_?axpy](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-axpy).
    template <class C, class T, std::ptrdiff_t DimX, std::ptrdiff_t DimY>
    TCM_SWARM_FORCEINLINE auto operator()(C const alpha,
        gsl::span<T const, DimX> const x, gsl::span<T, DimY> const y) const
        noexcept(!::tcm::detail::gsl_can_throw()) -> void
    {
        Expects(x.size() == y.size());
        constexpr difference_type one{1};
        return detail::axpy(gsl::narrow<difference_type>(x.size()), alpha,
            x.data(), one, y.data(), one);
    }
};

/// \brief Global instance of #tcm::mkl::axpy_fn.
TCM_SWARM_INLINE_VARIABLE(axpy_fn, axpy)

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_AXPY_HPP

