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

#ifndef TCM_SWARM_DETAIL_GEMV_HPP
#define TCM_SWARM_DETAIL_GEMV_HPP

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

struct gemv_fn {
    template <class T>
    TCM_SWARM_FORCEINLINE auto operator()(Layout const layout,
        Transpose const trans_a, size_type const m, size_type const n,
        T const alpha, T const* const a, difference_type const ld_a,
        T const* const x, difference_type const inc_x, T const beta,
        T* y, difference_type const inc_y) const noexcept -> void
    {
        return detail::gemv(to_raw_enum(layout), to_raw_enum(trans_a),
            static_cast<difference_type>(m),
            static_cast<difference_type>(n), alpha, a, ld_a, x, inc_x,
            beta, y, inc_y);
    }
};

/// \brief Computes matrix-vector product.
TCM_SWARM_INLINE_VARIABLE(gemv_fn, gemv);

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_GEMV_HPP

