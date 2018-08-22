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

#include "metropolis_local.hpp"
#include "detail/compact_index.hpp"
#include "detail/errors.hpp"
#include "detail/use_different_spin.hpp"
#include "logging.hpp"
#include "mcmc_config.hpp"
#include "mcmc_state.hpp"
#include "random.hpp"
#include "spin_utils.hpp"
#include <gsl/gsl>
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <optional>
#include <random>
#include <variant>
#include <vector>

TCM_SWARM_BEGIN_NAMESPACE

/// \rst
/// Generates a sequence of random flips.
/// \endrst
template <class Generator, std::ptrdiff_t Extent>
class RandomFlipper
    : private detail::CompactIndex<McmcState::index_type, Extent> {

    using number_flips_type =
        detail::CompactIndex<McmcState::index_type, Extent>;

  public:
    using typename number_flips_type::index_type;
    using value_type = gsl::span<index_type, Extent>;

    static_assert(Extent != 0,
        "RandomFlipper proposing 0 spin-flips at a time makes little sense.");

  private:
    constexpr auto number_flips() const noexcept -> index_type
    {
        return this->index();
    }

    auto validate_number_of_flips() const -> void
    {
        if (number_flips() < 0) {
            auto const msg =
                format(fmt("Requested a negative number of spin-flips: {}"),
                    number_flips());
            global_logger()->error(msg);
            throw_with_trace(std::invalid_argument{"Invalid argument"});
        }
        else if (number_flips() > _indices.size()) {
            auto const msg =
                format(fmt("Requested number of spin-flips exceeds the number "
                           "of spins in the system: {} > {}."),
                    number_flips(), _indices.size());
            global_logger()->error(msg);
            throw_with_trace(std::invalid_argument{"Invalid argument"});
        }
    }

  public:
    /// \defgroup RandomFlipperCopyMove Copy and Move
    /// \{
    RandomFlipper(RandomFlipper const&) = default;
    RandomFlipper(RandomFlipper&&)      = default;
    RandomFlipper& operator=(RandomFlipper const&) = default;
    RandomFlipper& operator=(RandomFlipper&&) = default;
    /// \}

    /// \brief Creates a new "flipper".
    RandomFlipper(Generator& generator, gsl::span<index_type> const workspace,
        number_flips_type const flips)
        : number_flips_type{flips}
        , _indices{workspace}
        , _generator{std::addressof(generator)}
        , _i{0}
    {
        using std::begin, std::end;
        validate_number_of_flips();
        Ensures(_i + number_flips() <= _indices.size());
        std::iota(begin(_indices), end(_indices), 0);
        shuffle();
    }

  private:
    auto shuffle()
    {
        using std::begin, std::end;
        std::shuffle(begin(_indices), end(_indices), *_generator);
    }

  public:
    auto read() const noexcept(!detail::gsl_can_throw()) -> value_type
    {
        Expects(_i + number_flips() <= _indices.size());
        return {_indices.data() + _i, number_flips()};
    }

    auto next(bool const /*unused*/) -> void
    {
        Expects(_i + number_flips() <= _indices.size());
        _i += number_flips();
        if (_i + number_flips() > _indices.size()) {
            shuffle();
            _i = 0;
        }
        Ensures(_i + number_flips() <= _indices.size());
    }

  private:
    gsl::span<index_type>
               _indices; ///< Workspace containing indices of spins we suggest flipping.
    gsl::not_null<Generator*> _generator; ///< `UniformRandomBitGenerator`_ to use.
    index_type
        _i; ///< Position in the _indices array indicating where to read the next proposal from.
};

template <class Generator, std::ptrdiff_t Extent>
class RandomFlipperWithMagnetisation {

  public:
    using index_type = McmcState::index_type;
    using value_type = gsl::span<index_type, Extent>;

    static_assert(Extent != 0, "RandomFlipperWithMagnetisation proposing 0 "
                               "spin-flips at a time makes little sense.");
    // static_assert(Extent == gsl::dynamic_extent || Extent % 2 == 0,
    //     "To preserve magnetisation number of spin-flips must be even.");

  public:
    /// \defgroup RandomFlipperWithMagnetisationCopyMove Copy and Move
    /// \{
    // clang-format off
    RandomFlipperWithMagnetisation(RandomFlipperWithMagnetisation const&) = delete;
    RandomFlipperWithMagnetisation(RandomFlipperWithMagnetisation&&) = default;
    RandomFlipperWithMagnetisation& operator=(RandomFlipperWithMagnetisation const&) = delete;
    RandomFlipperWithMagnetisation& operator=(RandomFlipperWithMagnetisation&&) = default;
    // clang-format on
    /// \}

  private:
    template <class C>
    auto check_arguments(gsl::span<C> const            initial_spin,
        gsl::span<index_type>                          workspace,
        detail::CompactIndex<index_type, Extent> const number_flips) const
    {
        Expects(is_valid_spin(initial_spin));
        if (number_flips.index() < 0) {
            auto const msg =
                format(fmt("Requested a negative number of spin-flips: {}"),
                    number_flips.index());
            global_logger()->error(msg);
            throw_with_trace(std::invalid_argument{"Invalid argument."});
        }
        else if (number_flips.index() > initial_spin.size()) {
            auto const msg =
                format(fmt("Requested number of spin-flips exceeds the number "
                           "of spins in the system: {} > {}."),
                    number_flips.index(), initial_spin.size());
            global_logger()->error(msg);
            throw_with_trace(std::invalid_argument{"Invalid argument"});
        }
        else if (number_flips.index() % 2 != 0) {
            auto const msg = format(
                fmt("Requested number of spin-flips is odd: {}. Can't preserve "
                    "magnetisation with an odd number of spin-flips."),
                number_flips.index());
            global_logger()->error(msg);
            throw_with_trace(std::invalid_argument{"Invalid argument"});
        }
        if (workspace.size() != initial_spin.size() + number_flips.index()) {
            auto const msg = format(
                fmt("Oh boy! Workspace has invalid size: {} != {} + {}. This "
                    "is definitely a bug. Please, report it to {}."),
                workspace.size(), initial_spin.size(), number_flips.index(),
                TCM_SWARM_ISSUES_LINK);
            global_logger()->critical(msg);
            throw_with_trace(std::invalid_argument{"BUG!"});
        }
    }

  public:
    /// \brief Creates a new "flipper".
    template <class C>
    RandomFlipperWithMagnetisation(gsl::span<C> const initial_spin,
        Generator& generator, gsl::span<index_type> workspace,
        detail::CompactIndex<index_type, Extent> const number_flips)
        : _proposed{workspace.data(), number_flips.index()}
        , _generator{std::addressof(generator)}
        , _i{0}
    {
        using std::begin, std::end;
        check_arguments(initial_spin, workspace, number_flips);

        _proposed = value_type{workspace.data(), number_flips.index()};
        std::fill(begin(_proposed), end(_proposed), -1);

        auto const begin_ups = begin(workspace) + number_flips.index();
        std::iota(begin_ups, end(workspace), 0);
        auto const begin_downs = std::partition(begin_ups, end(workspace),
            [s = initial_spin](auto const i) { return s[i] == C{1}; });
        _ups   = workspace.subspan(number_flips.index(), begin_downs - begin_ups);
        _downs = workspace.subspan(
            begin_downs - begin(workspace), end(workspace) - begin_downs);

        if (_ups.size() < _proposed.size() / 2
            || _downs.size() < _proposed.size() / 2) {
            auto const msg =
                format(fmt("Initial spin is invalid. Given {} spins up and {} "
                           "spins down, it's impossible to perform {} "
                           "spin-flips and still preserve the magnetisation."),
                    _ups.size(), _downs.size(), number_flips.index());
            global_logger()->error(msg);
            throw_with_trace(std::invalid_argument{"Invalid argument."});
        }
        shuffle();
        refill_start();
    }

  private:
    auto shuffle()
    {
        using std::begin, std::end;
        std::shuffle(begin(_ups), end(_ups), *_generator);
        std::shuffle(begin(_downs), end(_downs), *_generator);
    }

    auto refill_start() noexcept(!detail::gsl_can_throw())
    {
        using std::begin, std::end;
        auto const n = _proposed.size() / 2;
        Expects(_i + n <= _ups.size() && _i + n <= _downs.size());
        std::copy_n(begin(_ups) + _i, n, begin(_proposed));
        std::copy_n(begin(_downs) + _i, n, begin(_proposed) + n);
    }

    constexpr auto swap_accepted() noexcept(!detail::gsl_can_throw())
    {
        auto const n = _proposed.size() / 2;
        Expects(_i + n <= _ups.size() && _i + n <= _downs.size());
        for (auto i = _i; i < _i + n; ++i) {
            std::swap(_ups[i], _downs[i]);
        }
    }

  public:
    auto read() const noexcept -> value_type { return _proposed; }

    auto next(bool const accepted) -> void
    {
        auto const n = _proposed.size() / 2;
        Expects(_i + n <= _ups.size() && _i + n <= _downs.size());
        if (accepted) { swap_accepted(); }
        _i += n;
        if (_i + n > _ups.size() || _i + n > _downs.size()) {
            shuffle();
            _i = 0;
        }
        refill_start();
        Ensures(_i + n <= _ups.size() && _i + n <= _downs.size());
    }

  private:
    gsl::span<index_type>     _ups;
    gsl::span<index_type>     _downs;
    value_type                _proposed;
    gsl::not_null<Generator*> _generator;
    index_type                _i;
};

template <class Generator, std::ptrdiff_t Extent, class C>
RandomFlipperWithMagnetisation(gsl::span<C> initial_spin,
    Generator& generator, gsl::span<McmcState::index_type> workspace,
    detail::CompactIndex<McmcState::index_type, Extent> number_flips)
    -> RandomFlipperWithMagnetisation<Generator, Extent>;

/// \rst
/// For a given number of spins, an `UniformRandomBitGenerator`_, and a number
/// of spin-flips to propose, generates an infinite `ForwardRange`_ of
/// "proposals". Each proposal is a ``gsl::span<int const>`` containing indices
/// of spins that one should consider flipping.
/// \endrst

template <class Generator, std::ptrdiff_t Extent>
class MetropolisLocal {

  public:
    static_assert(Extent == gsl::dynamic_extent || Extent > 0,
        "Invalid number of flips.");

    using State             = McmcState;
    using C                 = State::C;
    using R                 = State::R;
    using index_type        = State::index_type;
    using number_flips_type = detail::CompactIndex<index_type, Extent>;

  private:
    using flipper_without = RandomFlipper<Generator, Extent>;
    using flipper_with    = RandomFlipperWithMagnetisation<Generator, Extent>;
    using flipper_type    = std::variant<flipper_without, flipper_with>;

    MetropolisLocal(State& state, Generator& generator,
        flipper_type&& flipper, std::vector<index_type>&& workspace)
        // clang-format off
            noexcept(
                std::is_nothrow_move_constructible_v<std::vector<index_type>>
             && std::is_nothrow_move_constructible_v<flipper_type>)
        // clang-format on
        : _state{std::addressof(state)}
        , _generator{std::addressof(generator)}
        , _flipper{std::move(flipper)}
        , _workspace{std::move(workspace)}
    {
    }

  public:
    template <class G, std::ptrdiff_t N>
    friend auto make_metropolis_local(
        McmcState&, G&, detail::CompactIndex<McmcState::index_type, N>, bool)
        -> MetropolisLocal<G, N>;

    MetropolisLocal(MetropolisLocal const&) = default;
    MetropolisLocal(MetropolisLocal&&)      = default;
    MetropolisLocal& operator=(MetropolisLocal const&) = default;
    MetropolisLocal& operator=(MetropolisLocal&&) = default;

    constexpr auto const& read() const noexcept { return *_state; }

    auto next() -> void
    {
        auto const u =
            std::generate_canonical<R, std::numeric_limits<R>::digits>(
                *_generator);
        auto const flips = std::visit(
            [](auto const& x) noexcept { return x.read(); }, _flipper);
        auto const [log_quot_wf, cache] = _state->log_quot_wf(flips);
        auto const probability =
            std::min(R{1}, std::norm(std::exp(log_quot_wf)));
        if (u <= probability) {
            _state->update(flips, cache);
            std::visit([](auto& x) { x.next(true); }, _flipper);
        }
        else {
            std::visit([](auto& x) { x.next(false); }, _flipper);
        }
    }

  private:
    gsl::not_null<State*>     _state;
    gsl::not_null<Generator*> _generator;
    flipper_type              _flipper;
    std::vector<index_type>   _workspace;
};

#if 0 // Using a factory function instead
template <class Generator, std::ptrdiff_t Extent>
MetropolisLocal(McmcState& state, Generator& generator,
    detail::CompactIndex<McmcState::index_type, Extent> number_flips,
    bool preserve_magnetisation) -> MetropolisLocal<Generator, Extent>;
#endif

template <class Generator, std::ptrdiff_t Extent>
inline auto make_metropolis_local(McmcState& state, Generator& generator,
    detail::CompactIndex<McmcState::index_type, Extent> number_flips,
    bool preserve_magnetisation) -> MetropolisLocal<Generator, Extent>
{
    if constexpr (Extent == gsl::dynamic_extent) {
        if (number_flips.index() < 0) {
            std::ostringstream msg;
            msg << "Requested number of spin-flips is negative ("
                << number_flips.index() << ").";
            throw_with_trace(std::invalid_argument{msg.str()});
        }
    }
    using T               = MetropolisLocal<Generator, Extent>;
    using index_type      = typename T::index_type;
    using flipper_type    = typename T::flipper_type;
    using flipper_with    = typename T::flipper_with;
    using flipper_without = typename T::flipper_without;

    if (preserve_magnetisation) {
        auto const workspace_size = gsl::narrow_cast<std::size_t>(
            state.size_visible() + number_flips.index());
        std::vector<index_type> workspace(workspace_size);
        flipper_type flipper{std::in_place_type<flipper_with>, state.spin(),
            generator, gsl::make_span(workspace), number_flips};
        return T{state, generator, std::move(flipper), std::move(workspace)};
    }
    else {
        auto const workspace_size =
            gsl::narrow_cast<std::size_t>(state.size_visible());
        std::vector<index_type> workspace(workspace_size);
        flipper_type flipper{std::in_place_type<flipper_without>, generator,
            gsl::make_span(workspace), number_flips};
        return T{state, generator, std::move(flipper), std::move(workspace)};
    }
}

namespace detail {

// clang-format off
template <std::ptrdiff_t Extent, class Accumulator, class Generator>
TCM_SWARM_FORCEINLINE
auto local_metropolis_loop_impl(McmcState& state, Accumulator& acc,
    MetropolisConfig const& config, Generator& gen) -> void
// clang-format on
{
    static_assert(Extent == gsl::dynamic_extent || Extent > 0,
        "Invalid number of spin-flips.");
    // Assume we can record at least one sample.
    Expects(config.steps().stop() > config.steps().start());

    auto const loop =
        [start = config.steps().start(), stop = config.steps().stop(),
            step = config.steps().step()](auto&& skip, auto&& record) {
            Steps::index_type i = 0;
            // Skipping [0, 1, ..., start)
            for (; i < start; ++i) {
                skip();
            }
            // Record [start]
            record();
            for (i += step; i < stop; i += step) {
                for (Steps::index_type j = 0; j < step; ++j) {
                    skip();
                }
                record();
            }
        };

    using compact_index = detail::CompactIndex<McmcState::index_type, Extent>;
    auto stream = make_metropolis_local(state, gen, compact_index{config.flips()},
        config.magnetisation().has_value());

    auto skip   = [&stream]() { stream.next(); };
    auto record = [&acc, &stream]() { std::invoke(acc, stream.read()); };
    loop(std::move(skip), std::move(record));
}

template <class Accumulator, class Generator>
auto local_metropolis_loop(McmcState& state, Accumulator& acc,
    MetropolisConfig const& config, Generator& gen) -> void
{
    using index_type = McmcState::index_type;
    auto const do_call = [&](auto n) {
        detail::local_metropolis_loop_impl<decltype(n)::value>(
            state, acc, config, gen);
    };
    switch (config.flips()) {
    case 1: do_call(std::integral_constant<index_type, 1>{}); break;
    case 2: do_call(std::integral_constant<index_type, 2>{}); break;
    case 3: do_call(std::integral_constant<index_type, 3>{}); break;
    case 4: do_call(std::integral_constant<index_type, 4>{}); break;
    case 5: do_call(std::integral_constant<index_type, 5>{}); break;
    case 6: do_call(std::integral_constant<index_type, 6>{}); break;
    case 7: do_call(std::integral_constant<index_type, 7>{}); break;
    case 8: do_call(std::integral_constant<index_type, 8>{}); break;
    case 9: do_call(std::integral_constant<index_type, 9>{}); break;
    case 10: do_call(std::integral_constant<index_type, 10>{}); break;
    default:
        do_call(std::integral_constant<index_type, gsl::dynamic_extent>{});
        break;
    } // end switch
}

} // namespace detail

auto sequential_local_metropolis(McmcState& state,
    std::function<void(McmcState const&)> acc, MetropolisConfig const& config,
    RandomGenerator& gen, RestartCallback restart_callback) -> void
{
    if (config.magnetisation().has_value()) {
        // TODO(twesterhout): I'm not sure whether this check actually belongs here.
        if (state.size_visible() == std::abs(config.magnetisation().value())) {
            acc(state);
            return;
        }
    }
    auto restarts = config.restarts();
    bool done     = false;
    do {
        try {
            detail::local_metropolis_loop(state, acc, config, gen);
            done = true;
        }
        catch (use_different_spin& e) {
            if (--restarts >= 0) { restart_callback(e.flips()); }
            else {
                global_logger()->error(
                    "Local energy could not be computed, and the allowed "
                    "number of restarts have already been performed. "
                    "Terminating this Monte-Carlo run.");
                done = true;
            }
        }
    } while (TCM_SWARM_UNLIKELY(!done));
}

#if 0
template <class Accumulator,
    class Generator = std::decay_t<decltype(thread_local_generator())>>
decltype(auto) sequential_local_metropolis(Rbm const& rbm, Accumulator&& acc,
    MC_Config const& config, Generator& gen = thread_local_generator())
{
    auto const state = rbm.make_state();
    McmcBase<typename Rbm::value_type> state =
        config.magnetisation().has_value()
            ? McmcBase{rbm, gen, *config.magnetisation()}
            : McmcBase{rbm, gen};
    if (config.magnetisation().has_value()) {
        // TODO: I'm not sure whether this check actually belongs here.
        if (state.size_visible() == std::abs(config.magnetisation().value())) {
            std::invoke(acc, state);
            return std::forward<Accumulator>(acc);
        }
    }
    return detail::local_metropolis_with_restarts(
        state, std::forward<Accumulator>(acc), config, gen);
}
#endif

TCM_SWARM_END_NAMESPACE

// #endif // TCM_SWARM_METROPOLIS_LOCAL_HPP

