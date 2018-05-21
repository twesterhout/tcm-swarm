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

template <std::size_t Dimension>
struct Heisenberg;

/// \brief Heisenberg Hamiltonian for spin-1/2 particles in one dimension.
template <>
struct Heisenberg<1u> {
  private:
    bool _periodic; ///< Whether to use periodic boundary conditions.

  public:
    explicit Heisenberg(bool periodic = false) noexcept
        : _periodic{periodic}
    {
    }

    template <class State>
    auto operator()(State const& state) const
    {
        using C           = typename State::value_type;
        using R           = typename C::value_type;
        using index_type  = typename State::index_type;
        using vector_type = typename State::vector_type;
        auto const spin   = state.spin();
        if (spin.size() <= 1) { return C{}; }
        using size_type = typename decltype(spin)::size_type;
        constexpr auto            cutoff = R{5};
        C                         energy{0};
        std::array<index_type, 2> flips;
        for (index_type i = 0, count = spin.size() - 1; i < count; ++i) {
            if (spin[i] == spin[i + 1]) { energy += C{1}; }
            else {
                flips = {i, i + 1};
                auto [log_quot_wf, cache] =
                    state.log_quot_wf(gsl::make_span(flips));
                if (log_quot_wf.real() >= cutoff) {
                    vector_type s{std::begin(spin), std::end(spin)};
                    s[i] *= C{-1};
                    s[i + 1] *= C{-1};
                    throw use_different_spin{std::move(s)};
                }
                energy += C{-1} + C{2} * std::exp(log_quot_wf);
            }
        }
        if (_periodic) {
            if (spin[0] == spin[spin.size() - 1]) { energy += C{1}; }
            else {
                flips = {0, spin.size() - 1};
                auto [log_quot_wf, crap] =
                    state.log_quot_wf(gsl::span<index_type const>{flips});
                if (log_quot_wf.real() >= cutoff) {
                    vector_type s{std::begin(spin), std::end(spin)};
                    s[0] *= C{-1};
                    s[spin.size() - 1] *= C{-1};
                    throw use_different_spin{std::move(s)};
                }
                energy += C{-1} + C{2} * std::exp(log_quot_wf);
            }
        }
        return energy;
    }
};

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_HEISENBERG_HPP
