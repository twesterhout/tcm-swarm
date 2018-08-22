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

#include "heisenberg.hpp"
#include "detail/errors.hpp"
#include "detail/use_different_spin.hpp"
#include "logging.hpp"
#include <tl/expected.hpp>
#include <omp.h>

_tcm_Heisenberg::_tcm_Heisenberg(std::vector<edge_type> edges,
    std::array<int, 2> const
        num_threads) noexcept(!tcm::detail::gsl_can_throw())
    : _edges{std::move(edges)}, _num_threads{num_threads}, _cutoff{std::nullopt}
{
    Expects(num_threads[0] > 0 && num_threads[1] > 0);
}

// clang-format off
_tcm_Heisenberg::_tcm_Heisenberg(std::vector<edge_type> edges,
    float const cutoff, std::array<int, 2> const num_threads) noexcept(
        !tcm::detail::gsl_can_throw())
    // clang-format on
    : _edges{std::move(edges)}, _num_threads{num_threads}, _cutoff{cutoff}
{
}

namespace {
struct kernel_fn {
  private:
    using State      = _tcm_Heisenberg::State;
    using C          = State::C;
    using R          = State::R;
    using index_type = State::index_type;

  public:
    TCM_SWARM_FORCEINLINE
    auto operator()(State const& state, C const* const spin,
        gsl::span<index_type const, 2> const flips) const noexcept -> C
    {
        return tcm::should_not_throw([&state, spin, flips]() {
            auto const i = flips[0];
            auto const j = flips[1];
            if (spin[i] == spin[j]) { return C{1}; }
            else {
                auto const [log_quot_wf, _ignored_] = state.log_quot_wf(flips);
                return C{-1} + C{2} * std::exp(log_quot_wf);
            }
        });
    }

    TCM_SWARM_FORCEINLINE
    auto operator()(State const& state, C const* const spin,
        gsl::span<index_type const, 2> const flips, R const cutoff) const
        noexcept -> tl::expected<C, std::array<index_type, 2>>
    {
        using result_type = tl::expected<C, std::array<index_type, 2>>;
        return tcm::should_not_throw([&state, spin, flips,
                                         cutoff]() -> result_type {
            auto const i = flips[0];
            auto const j = flips[1];
            if (spin[i] == spin[j]) { return C{1}; }
            else {
                auto const [log_quot_wf, _ignored_] = state.log_quot_wf(flips);
                if (log_quot_wf.real() > cutoff) {
                    return tl::make_unexpected(std::array{i, j});
                }
                return C{-1} + C{2} * std::exp(log_quot_wf);
            }
        });
    }
};

template <class T, class E>
TCM_SWARM_FORCEINLINE
constexpr auto increment_energy(tl::expected<T, E> acc, tl::expected<T, E> x)
{
    return x.and_then([acc](auto x_value) {
        return acc.map(
            [x_value](auto acc_value) { return acc_value + x_value; });
    });
}
} // unnamed namespace

auto _tcm_Heisenberg::operator()(State const& state) const
    -> C
{
    auto const* spin  = state.spin().data();
    if (_cutoff.has_value()) {
        using result_type = tl::expected<C, std::array<index_type, 2>>;
#pragma omp declare reduction(Energy:result_type                               \
                              : omp_out = increment_energy(omp_out, omp_in))
        auto const  cutoff = static_cast<C::value_type>(*_cutoff);
        result_type energy = C{0};
// clang-format off
#pragma omp parallel num_threads(_num_threads[0]) \
                     default(none) \
                     firstprivate(spin, cutoff) \
                     shared(state) \
                     reduction(Energy:energy)
        // clang-format on
        {
            // TODO(twesterhout): Maybe special case _num_threads[1] == 1?
            omp_set_num_threads(_num_threads[1]);
#pragma omp for
            for (std::size_t i = 0; i < _edges.size(); ++i) {
                // tcm::global_logger()->debug("i = {}, E = {} + {}i, {} <-> {}",
                //     i, energy.value().real(), energy.value().imag(),
                //     _edges[i][0], _edges[i][1]);
                energy = increment_energy(
                    energy, kernel_fn{}(state, spin, gsl::make_span(_edges[i]),
                                cutoff));
            }
        }

        if (energy.has_value()) {
            return *energy;
        }
        else {
            throw tcm::use_different_spin{energy.error()};
        }
    }
    else {
#pragma omp declare reduction(Energy:C : omp_out += omp_in)
        auto energy = C{0};
// clang-format off
#pragma omp parallel num_threads(_num_threads[0]) \
                     default(none) \
                     firstprivate(spin) \
                     shared(state) \
                     reduction(Energy:energy)
        // clang-format on
        {
            // TODO(twesterhout): Maybe special case _num_threads[1] == 1?
            omp_set_num_threads(_num_threads[1]);
#pragma omp for
            for (std::size_t i = 0; i < _edges.size(); ++i) {
                energy += kernel_fn{}(state, spin, gsl::make_span(_edges[i]));
            }
        }
        return energy;
    }
}

TCM_SWARM_BEGIN_NAMESPACE

namespace {
auto heisenberg_edges_1D(Heisenberg::index_type const n, bool const periodic)
    -> std::vector<Heisenberg::edge_type>
{
    if (n < 0) {
        global_logger()->error("System size can't be negative: {}.", n);
        throw_with_trace(std::invalid_argument{"Negative length."});
    }
    if (n < 2) {
        global_logger()->warn(
            "Creating a Heisenberg Hamiltonian for a system with {} spins. Are "
            "you sure that's what you want?",
            n);
        return {};
    }
    if (periodic && n == 2) {
        global_logger()->error(
            "Periodic boundary conditions of a system of 2 spins make little "
            "sense...");
        throw_with_trace(std::invalid_argument{"Invalid size."});
    }

    std::vector<Heisenberg::edge_type> edges;
    edges.reserve(static_cast<std::size_t>(n - 1 + periodic));
    for (Heisenberg::index_type i = 0; i < n - 1; ++i) {
        edges.emplace_back(std::array{i, i + 1});
    }
    if (periodic) { edges.emplace_back(std::array{0l, n - 1}); }
    return edges;
}
} // unnamed namespace

TCM_SWARM_SYMBOL_EXPORT
auto heisenberg_1D(Heisenberg::index_type const n, bool const periodic,
    std::array<int, 2> const num_threads) -> Heisenberg
{
    return Heisenberg{heisenberg_edges_1D(n, periodic), num_threads};
}

TCM_SWARM_SYMBOL_EXPORT
auto heisenberg_1D(Heisenberg::index_type const n, bool const periodic,
    float const cutoff, std::array<int, 2> const num_threads) -> Heisenberg
{
    return Heisenberg{heisenberg_edges_1D(n, periodic), cutoff, num_threads};
}

TCM_SWARM_END_NAMESPACE

