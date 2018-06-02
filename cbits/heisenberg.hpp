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

#ifndef TCM_SWARM_HEISENBERG_HPP
#define TCM_SWARM_HEISENBERG_HPP

#include <array>
#include <cmath>
#include <complex>
#include <utility>

#include <gsl/gsl>

#include "detail/config.hpp"
#include "detail/use_different_spin.hpp"

TCM_SWARM_BEGIN_NAMESPACE

template <class R, std::size_t Dimension, bool Periodic>
struct Heisenberg;

/// \brief Heisenberg Hamiltonian for spin-1/2 particles in one dimension.
template <class R, bool Periodic>
struct Heisenberg<R, 1u, Periodic> {
    constexpr Heisenberg() noexcept = default;
    constexpr Heisenberg(R const cutoff) noexcept : _cutoff{cutoff} {}
    constexpr Heisenberg(Heisenberg const&) = default;
    constexpr Heisenberg(Heisenberg&&) noexcept      = default;
    constexpr Heisenberg& operator=(Heisenberg const&) = default;
    constexpr Heisenberg& operator=(Heisenberg&&) noexcept = default;

    template <class State>
    auto operator()(State const& state) const -> typename State::value_type;

  private:
    // clang-format off
    template <class State, std::ptrdiff_t DimSpin>
    auto kernel(gsl::span<typename State::index_type const, 2> const flips,
        State const& state,
        gsl::span<typename State::value_type const, DimSpin> const spin) const
            -> typename State::value_type
    // clang-format on
    {
        using std::begin, std::end;
        using C           = typename State::value_type;
        using vector_type = typename State::vector_type;

        if (spin[flips[0]] == spin[flips[1]]) { return C{1}; }
        else {
            auto const [log_quot_wf, cache] = state.log_quot_wf(flips);
            if (_cutoff.has_value()) {
                if (log_quot_wf.real() > *_cutoff) {
                    vector_type s{begin(spin), end(spin)};
                    s[flips[0]] *= C{-1};
                    s[flips[1]] *= C{-1};
                    throw use_different_spin{std::move(s)};
                }
            }
            return C{-1} + C{2} * std::exp(log_quot_wf);
        }
    }

    std::optional<R> _cutoff;
};

template <class R, bool Periodic>
template <class State>
auto Heisenberg<R, 1u, Periodic>::operator()(State const& state) const ->
    typename State::value_type
{
    using C           = std::complex<R>;
    using index_type  = typename State::index_type;
    using vector_type = typename State::vector_type;
    static_assert(std::is_same_v<C, typename State::value_type>);

    auto const spin = state.spin();
    if (spin.size() <= 1) { return C{}; }
    using size_type = typename decltype(spin)::size_type;
    C                         energy{0};
    std::array<index_type, 2> flips;
    for (index_type i = 0, count = spin.size() - 1; i < count; ++i) {
        flips = {i, i + 1};
        energy += kernel(flips, state, spin);
    }
    if constexpr (Periodic) {
        flips = {0, spin.size() - 1};
        energy += kernel(flips, state, spin);
    }
    return energy;
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_HEISENBERG_HPP
