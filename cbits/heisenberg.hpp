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

#pragma once
#include "detail/use_different_spin.hpp"
#include "mcmc_state.hpp"
#include <gsl/gsl>
#include <array>
#include <omp.h>
#include <optional>

/// \brief Isotropic Heisenberg Hamiltonian for spin-1/2 particles.
struct TCM_SWARM_SYMBOL_VISIBLE _tcm_Heisenberg {
    using State      = tcm::McmcState;
    using C          = State::C;
    using index_type = State::index_type;
    using edge_type  = std::array<index_type, 2>;

    _tcm_Heisenberg() noexcept = default;
    _tcm_Heisenberg(std::vector<edge_type> edges,
        std::array<int, 2> num_threads) noexcept(!tcm::detail::gsl_can_throw());
    _tcm_Heisenberg(std::vector<edge_type> edges, float const cutoff,
        std::array<int, 2> num_threads) noexcept(!tcm::detail::gsl_can_throw());
    _tcm_Heisenberg(_tcm_Heisenberg const&)     = default;
    _tcm_Heisenberg(_tcm_Heisenberg&&) noexcept = default;
    _tcm_Heisenberg& operator=(_tcm_Heisenberg const&) = default;
    _tcm_Heisenberg& operator=(_tcm_Heisenberg&&) noexcept = default;

    auto operator()(State const& state) const -> C;

  private:
    std::vector<edge_type> _edges;
    std::array<int, 2>     _num_threads;
    std::optional<float>   _cutoff;
};

TCM_SWARM_BEGIN_NAMESPACE

using Heisenberg = _tcm_Heisenberg;

TCM_SWARM_SYMBOL_IMPORT
auto heisenberg_1D(Heisenberg::index_type const n, bool const periodic,
    std::array<int, 2> const num_threads = {omp_get_max_threads(), 1})
    -> Heisenberg;

// clang-format off
TCM_SWARM_SYMBOL_IMPORT
auto heisenberg_1D(Heisenberg::index_type const n, bool const periodic,
    float const cutoff, std::array<int, 2> const
    num_threads = {omp_get_max_threads(), 1}) -> Heisenberg;
// clang-format on

TCM_SWARM_END_NAMESPACE

