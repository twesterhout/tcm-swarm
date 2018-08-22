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

#ifndef TCM_SWARM_SPIN_UTILS_HPP
#define TCM_SWARM_SPIN_UTILS_HPP

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

    template <class R, class = std::enable_if_t<std::is_floating_point_v<R>>>
    constexpr auto operator()(std::complex<R> const x) const noexcept -> bool
    {
        return this->operator()(x.real()) && x.imag() == R{0};
    }

    template <class R>
    auto operator()(gsl::span<R> const x) const
        noexcept(!detail::gsl_can_throw()) -> bool
    {
        using std::cbegin, std::cend;
        return std::all_of(cbegin(x), cend(x),
            [](auto const s) -> bool { return is_valid_spin_fn{}(s); });
    }
};
} // namespace detail

/// \brief Checks whether a number or a span of numbers represents valid spins.
TCM_SWARM_INLINE_VARIABLE(detail::is_valid_spin_fn, is_valid_spin)

namespace detail {
struct magnetisation_fn {
  private:
    template <class R, class = std::enable_if_t<std::is_floating_point_v<R>>>
    static constexpr auto is_valid_magnetisation(R const x) noexcept -> bool
    {
        auto const i = gsl::narrow_cast<int>(x);
        return x == gsl::narrow_cast<R>(i);
    }

    template <class R, class = std::enable_if_t<std::is_floating_point_v<R>>>
    static constexpr auto is_valid_magnetisation(
        std::complex<R> const x) noexcept -> bool
    {
        return x.imag() == R{0} && is_valid_magnetisation(x.real());
    }

  public:
    template <class C>
    auto operator()(gsl::span<C> const x) const
        noexcept(!detail::gsl_can_throw())
    {
        using std::cbegin, std::cend;
        using index_type = typename gsl::span<C>::index_type;
        static_assert(
            std::is_signed_v<index_type>, "Magnetisation can be negative!");
        auto const m = std::accumulate(cbegin(x), cend(x), C{0});
        Expects(is_valid_magnetisation(m));
        return gsl::narrow_cast<index_type>(m.real());
    }
};
} // namespace detail


template <class C>
auto to_bitset(gsl::span<C> const spin) -> std::vector<bool>
{
    Expects(is_valid_spin(spin));
    using index_type = typename gsl::span<C>::index_type;
    static_assert(std::is_signed_v<index_type>);
    std::vector<bool> x(gsl::narrow_cast<std::size_t>(spin.size()));
    for (auto i = 0; i < spin.size(); ++i) {
        auto const s = spin[spin.size() - 1 - i];
        x[gsl::narrow_cast<std::size_t>(i)] = s == C{1};
    }
    return x;
}

#if 0
class CompactSpin {

    template <class C, std::ptrdiff_t Extent>
    auto to_byte(gsl::span<C, Extent> const x)
    {
        static_assert(1 <= Extent && Extent <= 8);
        std::byte b{x[Extent - 1] == 1};
        for (auto i = 2; i < Extent; ++i) {
          b |= std::byte{x[Extent - 1 - i] == 1} << 1;
        }
        return b;
    }

  private:
    std::array<char, 64> _storage;
    std::ptrdiff_t       _length;
};
#endif

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_SPIN_UTILS_HPP
