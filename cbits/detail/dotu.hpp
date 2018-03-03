// Copyright Tom Westerhout (c) 2017
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

#ifndef TCM_SWARM_DETAIL_UDOT_HPP
#define TCM_SWARM_DETAIL_UDOT_HPP

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace {

#define TCM_SWARM_DOTU_SIGNATURE_FOR(T)                              \
    auto dotu(difference_type const n, T const* const x,             \
        difference_type const inc_x, T const* const y,               \
        difference_type const inc_y) noexcept->T

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_PURE
        TCM_SWARM_DOTU_SIGNATURE_FOR(float)
        {
            return cblas_sdot(n, x, inc_x, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_PURE
        TCM_SWARM_DOTU_SIGNATURE_FOR(double)
        {
            return cblas_ddot(n, x, inc_x, y, inc_y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_PURE
        TCM_SWARM_DOTU_SIGNATURE_FOR(std::complex<float>)
        {
            std::complex<float> result;
            cblas_cdotu_sub(n, static_cast<void const*>(x), inc_x,
                static_cast<void const*>(y), inc_y,
                static_cast<void*>(&result));
            return result;
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_PURE
        TCM_SWARM_DOTU_SIGNATURE_FOR(std::complex<double>)
        {
            std::complex<double> result;
            cblas_zdotu_sub(n, static_cast<void const*>(x), inc_x,
                static_cast<void const*>(y), inc_y,
                static_cast<void*>(&result));
            return result;
        }

#undef TCM_SWARM_DOTU_SIGNATURE_FOR

    } // namespace
} // namespace detail

struct dotu_fn {
    // clang-format off
    template <class T>
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto operator()(size_type const n,
        T const* const x, difference_type const inc_x,
        T const* const y, difference_type const inc_y) const noexcept -> T
    {
        return detail::dotu(
            static_cast<difference_type>(n), x, inc_x, y, inc_y);
    }
    // clang-format on
};

/// \brief Computes unconjugated product of two vectors.
TCM_SWARM_INLINE_VARIABLE(dotu_fn, dotu)

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_UDOT_HPP

