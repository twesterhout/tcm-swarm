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

#ifndef TCM_SWARM_DETAIL_COPY_HPP
#define TCM_SWARM_DETAIL_COPY_HPP

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace {

#define TCM_SWARM_COPY_SIGNATURE_FOR(T)                              \
    auto copy(difference_type const n, T const* const x,             \
        difference_type const inc_x, T* const y,                     \
        difference_type const inc_y) noexcept->void

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_COPY_SIGNATURE_FOR(float)
        {
            cblas_scopy(n, x, inc_x, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_COPY_SIGNATURE_FOR(double)
        {
            cblas_dcopy(n, x, inc_x, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_COPY_SIGNATURE_FOR(std::complex<float>)
        {
            cblas_ccopy(n, x, inc_x, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_COPY_SIGNATURE_FOR(std::complex<double>)
        {
            cblas_zcopy(n, x, inc_x, y, inc_y);
        }

#undef TCM_SWARM_COPY_SIGNATURE_FOR
    } // namespace
} // namespace detail

struct copy_fn {
    template <class T>
    TCM_SWARM_FORCEINLINE auto operator()(size_type const n,
        T const* const x, difference_type const inc_x, T* const y,
        difference_type const inc_y) const noexcept -> void
    {
        return detail::copy(
            static_cast<difference_type>(n), x, inc_x, y, inc_y);
    }

    template <class T>
    TCM_SWARM_FORCEINLINE auto operator()(size_type const n,
        T const* const x, T* const y) const noexcept -> void
    {
        return detail::copy(static_cast<difference_type>(n), x,
            difference_type{1}, y, difference_type{1});
    }
};

/// \brief Copies first vector into the second.
TCM_SWARM_INLINE_VARIABLE(copy_fn, copy);

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_COPY_HPP

