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

// This file implements C++ wrappers around MKL's cblas_?scal functions.

#ifndef TCM_SWARM_DETAIL_SCALE_HPP
#define TCM_SWARM_DETAIL_SCALE_HPP

#include <gsl/span>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace {
#define TCM_SWARM_SCALE_SIGNATURE_FOR(C, T)                          \
    auto scale(difference_type n, C const alpha, T* const x,         \
        difference_type const inc_x) noexcept->void

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_SCALE_SIGNATURE_FOR(float, float)
        {
            cblas_sscal(n, alpha, x, inc_x);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_SCALE_SIGNATURE_FOR(double, double)
        {
            cblas_dscal(n, alpha, x, inc_x);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_SCALE_SIGNATURE_FOR(
            std::complex<float>, std::complex<float>)
        {
            cblas_cscal(n, &alpha, x, inc_x);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_SCALE_SIGNATURE_FOR(
            std::complex<double>, std::complex<double>)
        {
            cblas_zscal(n, &alpha, x, inc_x);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_SCALE_SIGNATURE_FOR(float, std::complex<float>)
        {
            cblas_csscal(n, alpha, x, inc_x);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_SCALE_SIGNATURE_FOR(double, std::complex<double>)
        {
            cblas_zdscal(n, alpha, x, inc_x);
        }

#undef TCM_SWARM_SCALE_SIGNATURE_FOR
    } // namespace
} // namespace detail

/// Let \f$V\f$ be a vector space over scalar field \f$\mathbb{F}\f$. Then
/// #scale: \f$\mathbb{F} \to V \to \{\ast\}\f$ is defined by:
/// \f$(\alpha, y) \mapsto y := \alpha y\f$.
struct scale_fn {
    /// \brief Implementation for #gsl::span.
    ///
    /// \see [cblas_?scal](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-scal)
    template <class C, class T>
    TCM_SWARM_FORCEINLINE auto operator()(
        C const alpha, gsl::span<T> const x) const
        noexcept(!::tcm::detail::gsl_can_throw()) -> void
    {
        return detail::scale(gsl::narrow<difference_type>(x.size()), alpha,
            x.data(), difference_type{1});
    }
};

/// \brief Global instance of #tcm::mkl::scale_fn.
TCM_SWARM_INLINE_VARIABLE(scale_fn, scale)

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_SCALE_HPP

