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

#ifndef TCM_SWARM_SPIN_HPP
#define TCM_SWARM_SPIN_HPP

#include <algorithm>
#include <complex>
#include <numeric>
#include <vector>

#include <gsl/span>

#include "detail/config.hpp"

TCM_SWARM_BEGIN_NAMESPACE

namespace detail {
struct is_valid_spin_fn {
    template <class T, class = std::enable_if_t<std::is_signed_v<T>>>
    constexpr auto operator()(T const x) const noexcept -> bool
    {
        return x == T{-1} || x == T{1};
    }

    template <class R,
        class = std::enable_if_t<std::is_floating_point_v<R>>>
    constexpr auto operator()(std::complex<R> const x) const noexcept
        -> bool
    {
        return this->operator()(x.real()) && x.imag() == R{0};
    }

    template <class R>
    auto operator()(gsl::span<R> const x) const
        noexcept(!detail::gsl_can_throw()) -> bool
    {
        using std::begin, std::end;
        return std::all_of(begin(x), end(x),
            [this](auto const s) -> bool { return this->operator()(s); });
    }
};
} // namespace detail

/// \brief Checks whether a number or a span of numbers represents valid spins.
TCM_SWARM_INLINE_VARIABLE(detail::is_valid_spin_fn, is_valid_spin)

namespace detail {
template <class R>
auto magnetisation(gsl::span<std::complex<R> const> const x) noexcept(
    !detail::gsl_can_throw())
{
    using std::begin, std::end;
    using C          = std::complex<R>;
    using index_type = typename gsl::span<C const>::index_type;
    auto const m     = std::accumulate(begin(x), end(x), C{0});
    Expects(m.imag() == R{0});
    return gsl::narrow<index_type>(m.real());
}
} // namespace detail

#if !defined(TCM_SWARM_NOCHECK_VALID_SPIN)
#define TCM_SWARM_IS_VALID_SPIN(x) ::TCM_SWARM_NAMESPACE::is_valid_spin(x)
#else
#define TCM_SWARM_IS_VALID_SPIN(x) true
#endif

#if !defined(TCM_SWARM_NOCHECK_MAGNETISATION)
#define TCM_SWARM_CHECK_MAGNETISATION(x, m)                               \
    (::TCM_SWARM_NAMESPACE::detail::magnetisation(x) == m)
#else
#define TCM_SWARM_CHECK_MAGNETISATION(x, m) true
#endif

template <class C>
auto to_bitset(gsl::span<C> const spin) -> std::vector<bool>
{
    using index_type = typename gsl::span<C>::index_type;
    static_assert(std::is_signed_v<index_type>);
    std::vector<bool> x(spin.size());
    for (auto i = 1; i <= spin.size(); ++i) {
        auto const s = spin[spin.size() - i];
        Expects(s == C{1} || s == C{-1});
        x[i - 1] = s == C{1};
    }
    return x;
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_SPIN_HPP

