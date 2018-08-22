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
#include "gradients.hpp"
#include "mcmc_state.hpp"
#include "metropolis_local.hpp"
#include "rbm_spin_float.hpp"
#include <gsl/span>
#include <functional>
#include <optional>

TCM_SWARM_BEGIN_NAMESPACE

TCM_SWARM_SYMBOL_IMPORT
auto sample_moments(Rbm const&              rbm,
    std::function<Rbm::C(McmcState const&)> hamiltonian,
    MetropolisConfig const& conf, gsl::span<Rbm::C> moments)
    -> std::tuple<Rbm::index_type, std::optional<Rbm::R>>;

TCM_SWARM_SYMBOL_IMPORT
auto sample_gradients(Rbm const&            rbm,
    std::function<Rbm::C(McmcState const&)> hamiltonian,
    MetropolisConfig const& config, gsl::span<Rbm::C> moments,
    gsl::span<Rbm::C> force, Gradients<Rbm::C> gradients)
    -> std::tuple<Rbm::index_type, std::optional<Rbm::R>>;

TCM_SWARM_END_NAMESPACE

