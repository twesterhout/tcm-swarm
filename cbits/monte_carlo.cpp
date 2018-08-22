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

#include "monte_carlo.hpp"
#include "accumulators.hpp"
#include "logging.hpp"
#include "metropolis_local.hpp"
#include "rbm_spin_float.hpp"
#include "spin_state.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <tl/expected.hpp>
#include <omp.h>

TCM_SWARM_BEGIN_NAMESPACE

namespace detail {
/// \brief Utility function to make a step from a run-time to a compile-time
/// integral constant.
template <class Func>
decltype(auto) using_intergral_constant(Func&& fn, int const n)
{
    Expects(n > 0);
    switch (n) {
    case 1: return std::forward<Func>(fn)(std::integral_constant<int, 1>{});
    case 2: return std::forward<Func>(fn)(std::integral_constant<int, 2>{});
    case 3: return std::forward<Func>(fn)(std::integral_constant<int, 3>{});
    case 4: return std::forward<Func>(fn)(std::integral_constant<int, 4>{});
    case 5: return std::forward<Func>(fn)(std::integral_constant<int, 5>{});
    case 6: return std::forward<Func>(fn)(std::integral_constant<int, 6>{});
    case 7: return std::forward<Func>(fn)(std::integral_constant<int, 7>{});
    case 8: return std::forward<Func>(fn)(std::integral_constant<int, 8>{});
    case 9: return std::forward<Func>(fn)(std::integral_constant<int, 9>{});
    case 10: return std::forward<Func>(fn)(std::integral_constant<int, 10>{});
    default: {
        auto const msg =
            format(fmt("Unfortunately, sampling of moments as high as {}'th is "
                       "not (yet) supported. If you need them, please, submit "
                       "an issue to {}"),
                n, TCM_SWARM_ISSUES_LINK);
        global_logger()->critical(msg);
        throw_with_trace(std::runtime_error{msg});
    }
    } // end switch
}

template <class Accumulator, class Continuation>
auto run_task(Rbm const& rbm, Accumulator& acc, MetropolisConfig const& config,
    Continuation&& merge)
{
    auto const state =
        rbm.make_state(config.magnetisation(), thread_local_generator());
    auto restart = [&acc, &state](auto&& flips) {
        global_logger()->error("An overflow encountered while computing the "
                               "local energy. Restarting the Monte-Carlo run.");
        state->update(std::forward<decltype(flips)>(flips), std::any{});
        acc.reset();
    };
    auto record = [&acc](auto&& x) { acc(std::forward<decltype(x)>(x)); };
    sequential_local_metropolis(*state, std::move(record), config,
        thread_local_generator(), std::move(restart));
    static_assert(noexcept(std::forward<Continuation>(merge)(acc)),
        "tcm::detail::run_task assumes that the continuation does not throw.");
    return std::forward<Continuation>(merge)(acc);
}

// clang-format off
template <std::ptrdiff_t Dim, class Hamiltonian, class Continuation>
TCM_SWARM_FORCEINLINE
auto run_moments_task(Rbm const& rbm, Hamiltonian hamiltonian,
    MetropolisConfig const& config, Continuation&& merge) noexcept
// clang-format on
    -> tl::expected<Rbm::index_type, std::exception_ptr>
{
    static_assert(Dim > 0);
    static_assert(std::is_nothrow_move_constructible_v<Hamiltonian>);
    using C                = Rbm::C;
    constexpr auto N       = gsl::narrow_cast<std::size_t>(Dim);
    using accumulator_type = CachingEnergyAccumulator<N, C, Hamiltonian>;
    try {
        accumulator_type acc{std::move(hamiltonian)};
        run_task(rbm, acc, config, std::forward<Continuation>(merge));
        return acc.cache().size();
    }
    catch (...) {
        return tl::make_unexpected(std::current_exception());
    }
}

// clang-format off
template <std::ptrdiff_t Dim, class Hamiltonian, class Continuation>
TCM_SWARM_FORCEINLINE
auto run_gradients_task(Rbm const& rbm, Hamiltonian hamiltonian,
    MetropolisConfig const& config, gsl::span<Rbm::C> const energies,
    Gradients<Rbm::C> gradients, Continuation&& merge) noexcept
// clang-format on
    -> tl::expected<Rbm::index_type, std::exception_ptr>
{
    static_assert(Dim > 0);
    static_assert(std::is_nothrow_move_constructible_v<Hamiltonian>);
    using C                = Rbm::C;
    constexpr auto N       = gsl::narrow_cast<std::size_t>(Dim);
    using accumulator_type = CachingGradientAccumulator<N, C, Hamiltonian>;
    try {
        accumulator_type acc{std::move(hamiltonian), energies, gradients};
        run_task(rbm, acc, config, std::forward<Continuation>(merge));
        return acc.cache().size();
    }
    catch (...) {
        return tl::make_unexpected(std::current_exception());
    }
}

template <std::ptrdiff_t N, class R>
TCM_SWARM_FORCEINLINE
auto axpy_static(R const alpha, gsl::span<std::complex<R> const, N> const x,
    R const beta, gsl::span<std::complex<R>, N> const y)
        noexcept(!detail::gsl_can_throw()) -> void
{
    static_assert(N >= 0);
    for (std::ptrdiff_t i = 0; i < N; ++i) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

template <std::ptrdiff_t Dim, class C, class = std::enable_if_t<(Dim > 0)>>
auto mean(
    gsl::span<std::array<C, gsl::narrow_cast<std::size_t>(Dim)> const> const xs,
    gsl::span<C, Dim> const out) noexcept(!detail::gsl_can_throw()) -> void
{
    // TODO(twesterhout): Strictly speaking, we should be doing binary
    // reduction...
    using R          = typename C::value_type;
    auto const scale = R{1} / gsl::narrow_cast<R>(xs.size());
    std::fill_n(out.data(), out.size(), C{0});
    for (auto const& x : xs) {
        axpy_static(scale, {x}, R{1}, out);
    }
}

template <class R>
auto are_close(R const x, R const y) noexcept -> bool
{
    static_assert(std::is_floating_point_v<R>);
    return std::abs(x - y) <= 2 * std::max(std::abs(x), std::abs(y))
                                  * std::numeric_limits<R>::epsilon();
}

// TODO(twesterhout): Yeah, this type signature is not very pretty...
template <std::ptrdiff_t Dim>
auto process_moments_results(
    std::vector<std::tuple<Rbm::index_type,
        std::array<Rbm::C, gsl::narrow_cast<std::size_t>(Dim)>>>&& results,
    gsl::span<Rbm::C, Dim> const                                   out)
    -> std::tuple<Rbm::index_type, std::optional<Rbm::R>>
{
    using std::begin, std::end;
    using C          = Rbm::C;
    using R          = Rbm::R;
    using index_type = Rbm::index_type;
    // using result_type = std::array<C, gsl::narrow_cast<std::size_t>(Dim)>;
    Expects(results.size() > 0);
    if (results.size() == 1) {
        auto const& [dim, result] = results.front();
        std::copy(begin(result), end(result), begin(out));
        global_logger()->warn(
            "Only one Monte-Carlo run performed: can't compute variance.");
        return {dim, std::nullopt};
    }
    else {
        // TODO(twesterhout): This is so inefficient...
        MomentsAccumulator<2, C> acc;
        std::for_each(begin(results), end(results), [&acc](auto const& _x) {
            auto const& x = std::get<1>(_x);
            SPDLOG_DEBUG(
                global_logger(), "E = {} + {}i.", x[0].real(), x[0].imag());
            acc(x[0]);
        });
        auto const variance =
            acc.template get<2>().real() + acc.template get<2>().imag();
        // TODO(twesterhout): Strictly speaking, we should be doing binary
        // reduction...
        auto const scale = R{1} / gsl::narrow_cast<R>(results.size());
        index_type dim   = 0;
        std::fill_n(out.data(), out.size(), C{0});
        for (auto const& [n, x] : results) {
            axpy_static(scale, {x}, R{1}, out);
            dim += n;
        }
        dim = gsl::narrow_cast<index_type>(
            std::round(scale * gsl::narrow_cast<R>(dim)));
        return {dim, {variance}};
    }
}

template <std::ptrdiff_t Dim, class Hamiltonian>
auto sample_moments(Rbm const& rbm, Hamiltonian&& hamiltonian,
    MetropolisConfig const& config, gsl::span<Rbm::C, Dim> const moments)
    -> std::tuple<Rbm::index_type, std::optional<Rbm::R>>
{
    if constexpr (Dim == gsl::dynamic_extent) {
        return detail::using_intergral_constant(
            [&](auto N) {
                return sample_moments<decltype(N)::value>(rbm,
                    std::forward<Hamiltonian>(hamiltonian), config, moments);
            },
            moments.size());
    }
    else if constexpr (Dim == 0) {
        return {0, std::nullopt};
    }
    else {
        static_assert(Dim > 0);
        Expects(config.runs() > 0);
        constexpr auto N  = gsl::narrow_cast<std::size_t>(Dim);
        using C           = Rbm::C;
        using result_type = MomentsAccumulator<N, C>;
        using index_type  = Rbm::index_type;
        std::exception_ptr       e_ptr;
        std::vector<std::tuple<index_type, std::array<C, N>>> results(
            gsl::narrow_cast<std::size_t>(config.runs()));
#pragma omp parallel for num_threads(config.threads()[0])
        for (Rbm::index_type i = 0; i < config.runs(); ++i) {
            SPDLOG_DEBUG(global_logger(),
                "Using a team of {} threads to sample moments.",
                omp_get_num_threads());
            auto out = gsl::span<C, Dim>{
                std::get<1>(results[gsl::narrow_cast<std::size_t>(i)])};
            auto dimension =
                detail::run_moments_task<Dim>(rbm, std::cref(hamiltonian),
                    config, [out](auto&& acc) noexcept { acc.copy_to(out); });
            if (dimension.has_value()) {
                std::get<0>(results[gsl::narrow_cast<std::size_t>(i)]) =
                    *dimension;
            }
            else {
#pragma omp critical
                {
                    global_logger()->error(
                        "Task #{} terminated with an exception.", i);
                    e_ptr = std::move(dimension).error();
                }
            }
        }
        if (e_ptr) { std::rethrow_exception(e_ptr); }
        return detail::process_moments_results(std::move(results), moments);
    }
}

// clang-format off
template <class C>
auto compute_force(gsl::span<C> const energies, C const mean_energy,
    Gradients<C> const derivatives, gsl::span<C> const out)
    noexcept(!detail::gsl_can_throw()) -> void
// clang-format on
{
    using R                      = typename C::value_type;
    using index_type             = typename gsl::span<C>::index_type;
    auto const number_steps      = energies.size();
    auto const number_parameters = derivatives.template extent<1>();
    Expects(derivatives.template extent<0>() == number_steps);
    Expects(out.size() == number_parameters);
    Expects(number_steps > 0 && number_parameters > 0);
    // E <- E - 〈E〉
    // TODO(twesterhout): Optimise this using non-temporal writes.
    auto* e_raw = energies.data();
    Expects(detail::is_aligned<alignment<R>()>(e_raw));
#pragma omp parallel for simd default(none) firstprivate(mean_energy)          \
    shared(e_raw) aligned(e_raw                                                \
                          : alignment <R>())
    for (index_type i = 0; i < number_steps; ++i) {
        e_raw[i] -= mean_energy;
    }
    // out <- 1/number_steps * O^H E
    mkl::gemv(mkl::Layout::RowMajor, mkl::Transpose::ConjTrans,
        derivatives.rows, derivatives.cols, C{1} / gsl::narrow<R>(number_steps),
        static_cast<C const*>(derivatives.data), derivatives.stride,
        energies.data(), 1, C{0}, out.data(), 1);
}

template auto compute_force(gsl::span<std::complex<float>>, std::complex<float>,
    Gradients<std::complex<float>>,
    gsl::span<std::complex<float>>) noexcept(!detail::gsl_can_throw()) -> void;

template auto compute_force(gsl::span<std::complex<double>>,
    std::complex<double>, Gradients<std::complex<double>>,
    gsl::span<std::complex<double>>) noexcept(!detail::gsl_can_throw()) -> void;


// TODO(twesterhout): This function is _way_ too long!
template <std::ptrdiff_t Dim, class Hamiltonian>
auto sample_gradients(Rbm const& rbm, Hamiltonian&& hamiltonian,
    MetropolisConfig const& config, gsl::span<Rbm::C, Dim> const moments,
    gsl::span<Rbm::C> const force, Gradients<Rbm::C> gradients)
    -> std::tuple<Rbm::index_type, std::optional<Rbm::R>>
{
    if constexpr (Dim == gsl::dynamic_extent) {
        return detail::using_intergral_constant(
            [&](auto N) {
                return sample_gradients<decltype(N)::value>(rbm,
                    std::forward<Hamiltonian>(hamiltonian), config, moments,
                    force, gradients);
            },
            moments.size());
    }
    else {
        // Shortcut #1: there's nothing to do.
        if (config.steps().count() == 0) { return {0, std::nullopt}; }
        // Shortcut #2: only a single spin configuration is possible.
        if (config.magnetisation().has_value()) {
            if (rbm.size_visible()
                == std::abs(config.magnetisation().value())) {
                auto const state = rbm.make_state(
                    config.magnetisation(), thread_local_generator());
                state->der_log_wf(gradients[0]);
                std::fill_n(force.data(), force.size(), 0);
                if constexpr (Dim > 0) {
                    moments[0] = hamiltonian(*state);
                    std::fill(std::begin(moments) + 1, std::end(moments), 0);
                }
                return {1, std::nullopt};
            }
        }
        // Sanity checks:
        auto const number_steps = config.steps().count();
        auto const n            = number_steps * config.runs();
        Expects(force.size() == rbm.size());
        Expects(gradients.template extent<0>() == n);
        Expects(gradients.template extent<1>() == rbm.size());
        constexpr auto N = gsl::narrow_cast<std::size_t>(Dim);
        using C          = Rbm::C;
        using index_type = Rbm::index_type;
        auto energies_storage =
            std::get<0>(Rbm::allocate_buffer(std::max(n, rbm.size())));
        auto energies = gsl::make_span(energies_storage.get(), n);
        std::vector<std::tuple<index_type, std::array<C, N>>> results(
            gsl::narrow_cast<std::size_t>(config.runs()));
        std::exception_ptr e_ptr;
#pragma omp parallel for num_threads(config.threads()[0])
        for (auto i = 0; i < config.runs(); ++i) {
            global_logger()->debug(
                "Using a team of {} threads to sample gradients.",
                omp_get_num_threads());
            auto out = gsl::span<C, Dim>{
                std::get<1>(results[gsl::narrow_cast<std::size_t>(i)])};
            auto dimension =
                detail::run_gradients_task<Dim>(rbm, std::cref(hamiltonian),
                    config, energies.subspan(i * number_steps, number_steps),
                    gradients.sub_rows(i * number_steps, number_steps),
                    [out](auto&& acc) noexcept { acc.copy_to(out); });
            if (dimension.has_value()) {
                std::get<0>(results[gsl::narrow_cast<std::size_t>(i)]) =
                    *dimension;
            }
            else {
#pragma omp critical
                {
                    global_logger()->error(
                        "Task #{} terminated with an exception.", i);
                    e_ptr = std::move(dimension).error();
                }
            }
        }
        if (e_ptr) { std::rethrow_exception(e_ptr); }
        auto stats =
            detail::process_moments_results(std::move(results), moments);
        auto const restore_omp_threads =
            gsl::finally([n = omp_get_max_threads()]() noexcept {
                SPDLOG_DEBUG(global_logger(),
                    "Restoring the number of OpenMP threads to {}.", n);
                omp_set_num_threads(n);
            });
        auto const total_num_threads =
            config.threads()[0] * config.threads()[1] * config.threads()[2];
        SPDLOG_DEBUG(global_logger(),
            "Temporary setting the number of OpenMP threads to {}.",
            total_num_threads);
        omp_set_num_threads(total_num_threads);
        detail::compute_force(energies, moments[0], gradients, force);
        reshift(gradients, energies.subspan(0, rbm.size()));
        return stats;
    }
}

} // namespace detail

/*
template auto sample_gradients(Rbm const&,
    std::function<Rbm::C(McmcState const&)>&, MetropolisConfig const&,
    gsl::span<Rbm::C>, gsl::span<Rbm::C>, Gradients<Rbm::C>) -> void;
*/

TCM_SWARM_SYMBOL_EXPORT
auto sample_moments(Rbm const&              rbm,
    std::function<Rbm::C(McmcState const&)> hamiltonian,
    MetropolisConfig const& conf, gsl::span<Rbm::C> moments)
    -> std::tuple<Rbm::index_type, std::optional<Rbm::R>>
{
    return detail::sample_moments(rbm, std::move(hamiltonian), conf, moments);
}

TCM_SWARM_SYMBOL_EXPORT
auto sample_gradients(Rbm const&            rbm,
    std::function<Rbm::C(McmcState const&)> hamiltonian,
    MetropolisConfig const& config, gsl::span<Rbm::C> moments,
    gsl::span<Rbm::C> force, Gradients<Rbm::C> gradients)
    -> std::tuple<Rbm::index_type, std::optional<Rbm::R>>
{
    return detail::sample_gradients(
        rbm, std::move(hamiltonian), config, moments, force, gradients);
}

TCM_SWARM_END_NAMESPACE
