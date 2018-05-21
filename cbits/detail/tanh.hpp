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

// This file implements C++ wrappers around MKL's cblas_?dot? functions.

#ifndef TCM_SWARM_DETAIL_TANH_HPP
#define TCM_SWARM_DETAIL_TANH_HPP

#include <gsl/gsl_assert>
#include <gsl/span>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
    namespace {

#define TCM_SWARM_TANH_SIGNATURE_FOR(T)                                   \
    auto tanh(difference_type const n, T const* const x,                  \
        T* const y) noexcept->void

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_TANH_SIGNATURE_FOR(float)
        {
            return vsTanh(n, x, y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_TANH_SIGNATURE_FOR(double)
        {
            return vdTanh(n, x, y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_TANH_SIGNATURE_FOR(std::complex<float>)
        {
            return vcTanh(n, x, y);
        }

        TCM_SWARM_FORCEINLINE
        TCM_SWARM_TANH_SIGNATURE_FOR(std::complex<double>)
        {
            return vzTanh(n, x, y);
        }

#undef TCM_SWARM_TANH_SIGNATURE_FOR

    } // namespace
} // namespace detail

struct tanh_fn {
    // clang-format off
    template <class T>
    TCM_SWARM_FORCEINLINE
    auto operator()(gsl::span<T const> const x, gsl::span<T> const y)
        const noexcept(!::TCM_SWARM_NAMESPACE::detail::gsl_can_throw())
    // clang-format on
    {
        Expects(x.size() == y.size());
        return detail::tanh(
            gsl::narrow<difference_type>(x.size()), x.data(), y.data());
    }
};

/// \brief Global instance of #tcm::mkl::tanh_fn.
TCM_SWARM_INLINE_VARIABLE(tanh_fn, tanh)

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_TANH_HPP

