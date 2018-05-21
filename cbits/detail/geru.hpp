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

// This file implements C++ wrappers around MKL's cblas_?ger? functions.

#ifndef TCM_SWARM_DETAIL_GERU_HPP
#define TCM_SWARM_DETAIL_GERU_HPP

#include <gsl/gsl>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace {

#define TCM_SWARM_GERU_SIGNATURE_FOR(T)                                   \
    auto geru(CBLAS_LAYOUT const layout, difference_type const m,         \
        difference_type const n, T const alpha, T const* const x,         \
        difference_type const inc_x, T const* const y,                    \
        difference_type const inc_y, T* const a,                          \
        difference_type const ld_a) noexcept->void

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_GERU_SIGNATURE_FOR(float)
        {
            return cblas_sger(
                layout, m, n, alpha, x, inc_x, y, inc_y, a, ld_a);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_GERU_SIGNATURE_FOR(double)
        {
            return cblas_dger(
                layout, m, n, alpha, x, inc_x, y, inc_y, a, ld_a);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_GERU_SIGNATURE_FOR(std::complex<float>)
        {
            cblas_cgeru(layout, m, n, &alpha, x, inc_x, y, inc_y, a, ld_a);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_GERU_SIGNATURE_FOR(std::complex<double>)
        {
            cblas_zgeru(layout, m, n, &alpha, x, inc_x, y, inc_y, a, ld_a);
        }

#undef TCM_SWARM_GERU_SIGNATURE_FOR

    } // namespace
} // namespace detail

struct geru_fn {
    template <class T>
    TCM_SWARM_FORCEINLINE auto operator()(T const alpha,
        gsl::span<T const> const x, gsl::span<T const> const y,
        gsl::span<T> const a, mkl::Layout const layout) const
        noexcept(!tcm::detail::gsl_can_throw()) -> void
    {
        Expects(x.size() * y.size() == a.size());
        constexpr difference_type one{1};
        auto const m    = gsl::narrow<difference_type>(x.size());
        auto const n    = gsl::narrow<difference_type>(y.size());
        auto const ld_a = layout == Layout::ColMajor ? m : n;
        return detail::geru(to_raw_enum(layout), m, n, alpha, x.data(),
            one, y.data(), one, a.data(), ld_a);
    }
};

/// \brief Global instance of #tcm::mkl::geru_fn.
TCM_SWARM_INLINE_VARIABLE(geru_fn, geru)

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_GERU_HPP

