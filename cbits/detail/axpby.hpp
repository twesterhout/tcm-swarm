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

#ifndef TCM_SWARM_DETAIL_AXPBY_HPP
#define TCM_SWARM_DETAIL_AXPBY_HPP

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace /*anonymous*/ {

#define TCM_SWARM_AXPBY_SIGNATURE_FOR(C, T)                          \
    auto axpby(difference_type n, C const alpha, T const* const x,   \
        difference_type const inc_x, C const beta, T* const y,       \
        difference_type const inc_y) noexcept->void

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_AXPBY_SIGNATURE_FOR(float, float)
        {
            cblas_saxpby(n, alpha, x, inc_x, beta, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_AXPBY_SIGNATURE_FOR(double, double)
        {
            cblas_daxpby(n, alpha, x, inc_x, beta, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_AXPBY_SIGNATURE_FOR(
            std::complex<float>, std::complex<float>)
        {
            cblas_caxpby(n, &alpha, x, inc_x, &beta, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_AXPBY_SIGNATURE_FOR(
            std::complex<double>, std::complex<double>)
        {
            cblas_zaxpby(n, &alpha, x, inc_x, &beta, y, inc_y);
        }

#undef TCM_SWARM_AXPBY_SIGNATURE_FOR
    } // namespace
} // namespace detail

struct axpby_fn {
    template <class C, class T>
    TCM_SWARM_FORCEINLINE auto operator()(size_type n, C const alpha,
        T const* const x, difference_type const inc_x, C const beta,
        T* const y, difference_type const inc_y) const noexcept
        -> void
    {
        return detail::axpby(static_cast<difference_type>(n), alpha,
            x, inc_x, beta, y, inc_y);
    }
};

/// \brief Performs the assignment `y := alpha * x + beta * y`
TCM_SWARM_INLINE_VARIABLE(axpby_fn, axpby);

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_AXPBY_HPP

