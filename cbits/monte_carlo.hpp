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

#ifndef TCM_SWARM_MONTE_CARLO_HPP
#define TCM_SWARM_MONTE_CARLO_HPP

#include <omp.h>

#include "accumulators.hpp"
#include "metropolis_local.hpp"

TCM_SWARM_BEGIN_NAMESPACE

template <class Rbm, class Hamiltonian, class Int>
auto sample_energy(Rbm const& rbm,         ///< Our psi
    Hamiltonian               hamiltonian, ///< Hamiltonian to sample
    Int const number_runs, ///< Number of Monte-Carlo runs to perform
    std::tuple<Int, Int, Int> const steps, ///< (start, step, stop)
    Int const number_flips, ///< Number of flips to perform at each step
    std::optional<Int> const
              magnetisation, ///< Sample states with a specific magnetisation
    Int const max_restarts = Int{1}, ///< Maximum number of restarts
    Int const number_threads = Int{omp_get_max_threads()}
    ///< Number of threads to use
) -> typename Rbm::value_type
{
    using C = typename Rbm::value_type;
    using result_type = EnergyAccumulator<1, C, Hamiltonian>;

#pragma omp declare reduction(Merge:result_type:omp_out.merge(omp_in)) \
    initializer(omp_priv{omp_orig})

    result_type result{hamiltonian};
#pragma omp parallel reduction(Merge:result)
    for (auto i = number_runs; i > 0; --i) {
        CachingEnergyAccumulator<1, C, Hamiltonian> acc{hamiltonian};
        sequential_local_metropolis(
            rbm, steps, acc, number_flips, magnetisation, max_restarts);
        result.merge(acc);
    }

    return result.template get<1>();
}

template <std::size_t N, class Rbm, class Hamiltonian, class Int>
auto sample_moments(Rbm const& rbm,         ///< Our psi
    Hamiltonian               hamiltonian, ///< Hamiltonian to sample
    Int const number_runs, ///< Number of Monte-Carlo runs to perform
    std::tuple<Int, Int, Int> const steps, ///< (start, step, stop)
    Int const number_flips, ///< Number of flips to perform at each step
    std::optional<Int> const
              magnetisation, ///< Sample states with a specific magnetisation
    Int const max_restarts = Int{1}, ///< Maximum number of restarts
    Int const number_threads = Int{omp_get_max_threads()}
    ///< Number of threads to use
) -> std::array<typename Rbm::value_type, N>
{
    static_assert(N >= 2);
    using C = typename Rbm::value_type;
    using result_type = EnergyAccumulator<N, C, Hamiltonian>;

#pragma omp declare reduction(Merge:result_type:omp_out.merge(omp_in)) \
    initializer(omp_priv{omp_orig})

    result_type result{hamiltonian};
#pragma omp parallel reduction(Merge:result)
    for (auto i = number_runs; i > 0; --i) {
        CachingEnergyAccumulator<N, C, Hamiltonian> acc{hamiltonian};
        sequential_local_metropolis(
            rbm, steps, acc, number_flips, magnetisation, max_restarts);
        result.merge(acc);
    }

    std::array<C, N> moments;

    static_assert(N == 4);
    moments[0] = result.template get<1>();
    moments[1] = result.template get<2>();
    moments[2] = result.template get<3>();
    moments[3] = result.template get<4>();
    return moments;
}

template <class Rbm, class Hamiltonian, class Int>
TCM_SWARM_NOINLINE
auto sample_gradients(Rbm const& rbm,         ///< Our psi
    Hamiltonian                  hamiltonian, ///< Hamiltonian to sample
    Int const number_runs, ///< Number of Monte-Carlo runs to perform
    std::tuple<Int, Int, Int> const steps, ///< (start, step, stop)
    Int const number_flips, ///< Number of flips to perform at each step
    std::optional<Int> const
              magnetisation, ///< Sample states with a specific magnetisation
    Int const max_restarts   = Int{1}, ///< Maximum number of restarts
    Int const number_threads = Int{omp_get_max_threads()}
    ///< Number of threads to use
)
{
    using C           = typename Rbm::value_type;
    using result_type = EnergyAccumulator<1, C, Hamiltonian>;
    using vector_type = typename Rbm::vector_type;

    auto const [low, step, high] = steps;
    auto const  number_steps     = (high - low - 1) / step + 1;
    vector_type energies_storage(number_runs * number_steps);
    vector_type gradients_storage(number_runs * number_steps * rbm.size());

    auto const energies  = gsl::make_span(energies_storage);
    auto const gradients = gsl::as_multi_span(gradients_storage.data(),
        gsl::dim(number_runs * number_steps), gsl::dim(rbm.size()));

#pragma omp declare reduction(Merge:result_type                           \
                              : omp_out.merge(omp_in))                    \
    initializer(omp_priv{omp_orig})

    result_type result{hamiltonian};
#pragma omp parallel for reduction(Merge : result)
    for (Int i = 0; i < number_runs; ++i) {
        auto es = energies.subspan(i * number_steps, number_steps);
        auto gs = gsl::as_multi_span(
            gradients.data() + i * number_steps * rbm.size(),
            gsl::dim(number_steps), gsl::dim(rbm.size()));
        GradientAccumulator<1, C, Hamiltonian> acc{hamiltonian, es, gs};
        sequential_local_metropolis(
            rbm, steps, acc, number_flips, magnetisation, max_restarts);
        result.merge(acc);
    }

    return std::make_tuple(result.template get<1>(), std::move(energies_storage),
        std::move(gradients_storage));
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_MONTE_CARLO_HPP
