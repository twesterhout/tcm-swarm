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

#ifndef TCM_SWARM_METROPOLIS_LOCAL_HPP
#define TCM_SWARM_METROPOLIS_LOCAL_HPP

#include "detail/config.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <complex>
#include <iostream> // TODO: Get rid of this.
#include <numeric>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include <gsl/gsl>

// Let's include range-v3
#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wpadded"
#elif defined(TCM_SWARM_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wnoexcept"
#endif
#include <range/v3/view/drop.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view_adaptor.hpp>
#include <range/v3/view_facade.hpp>
#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic pop
#elif defined(TCM_SWARM_GCC)
#pragma GCC diagnostic pop
#endif

#include "detail/random.hpp"
#include "detail/use_different_spin.hpp"
#include "rbm_spin.hpp"

TCM_SWARM_BEGIN_NAMESPACE

/// \rst
/// Generates a sequence of random flips.
/// \endrst
template <class Generator, class IndexType = int>
class RandomFlipper {

    using self_type = RandomFlipper<Generator, IndexType>;

  public:
    using index_type  = IndexType;
    using vector_type = std::vector<index_type>;
    using size_type   = typename vector_type::size_type;
    using value_type  = gsl::span<index_type>;

    static_assert(std::is_same_v<std::decay_t<Generator>, Generator>);
    static_assert(std::is_same_v<std::decay_t<IndexType>, IndexType>);
    static_assert(std::is_signed_v<IndexType>);

  private:
    gsl::span<index_type>
               _indices; ///< Workspace containing indices of spins we suggest flipping.
    Generator* _generator; ///< `UniformRandomBitGenerator`_ to use.
    index_type
        _number_flips; ///< Number of spin-flips to propose at a time.
    index_type
        _i; ///< Position in the _indices array indicating where to read the next proposal from.

  public:
#if 1
    ///< Default constructor
    constexpr RandomFlipper() noexcept
        : _indices{}
        , _generator{nullptr}
        , _number_flips{1}
    {
    }
#endif

    /// \defgroup RandomFlipperCopyMove Copy and Move
    /// \{
    RandomFlipper(RandomFlipper const&)     = delete;
    RandomFlipper(RandomFlipper&&) noexcept = default;
    RandomFlipper& operator=(RandomFlipper const&) = delete;
    RandomFlipper& operator=(RandomFlipper&&) noexcept = default;
    /// \}

    /// \brief Creates a new "flipper".
    // clang-format off
    template <class C>
    RandomFlipper(gsl::span<C> const initial_spin, Generator& generator,
        gsl::span<index_type> const workspace, index_type const number_flips = 1)
        // clang-format on
        : _indices{workspace}
        , _generator{std::addressof(generator)}
        , _number_flips{number_flips}
    {
        using std::begin, std::end;
        if (initial_spin.size() == 0) {
            throw_with_trace(std::invalid_argument{
                "Initial spin configuration has size 0."});
        }
        if (number_flips < 0) {
            std::ostringstream msg;
            msg << "Requested number of spin-flips is negative ("
                << number_flips << ").";
            throw_with_trace(std::invalid_argument{msg.str()});
        }
        if (number_flips > initial_spin.size()) {
            std::ostringstream msg;
            msg << "Requested number of spin-flips (" << number_flips
                << ") exceeds the number of spins (" << initial_spin.size()
                << ").";
            throw_with_trace(std::invalid_argument{msg.str()});
        }
        Expects(workspace.size() == initial_spin.size());
        std::iota(begin(_indices), end(_indices), 0);
        shuffle();
        Ensures(_generator != nullptr);
    }

  private:
    auto shuffle()
    {
        using std::begin, std::end;
        Expects(_generator != nullptr);
        std::shuffle(begin(_indices), end(_indices), *_generator);
    }

  public:
    auto read() const noexcept(!detail::gsl_can_throw())
        -> gsl::span<index_type const>
    {
        Expects(_generator != nullptr);
        Expects(_i + _number_flips <= _indices.size());
        return {_indices.data() + _i, _number_flips};
    }

    auto next(bool const /*unused*/) -> void
    {
        Expects(_generator != nullptr);
        Expects(_i + _number_flips <= _indices.size());
        _i += _number_flips;
        if (_i + _number_flips > _indices.size()) {
            shuffle();
            _i = 0;
        }
    }
};


template <class Generator, class IndexType = int>
class RandomFlipperWithMagnetisation {

  public:
    using index_type  = IndexType;
    using vector_type = std::vector<index_type>;
    using size_type   = typename vector_type::size_type;

    static_assert(std::is_signed_v<IndexType>);

  private:
    gsl::span<index_type> _ups;
    gsl::span<index_type> _downs;
    gsl::span<index_type> _proposed;
    Generator*            _generator;
    index_type            _i;

#if 0
    auto print_state()
    {
        using std::begin, std::end;
        std::cout << "[";
        std::copy(begin(_proposed), end(_proposed),
            std::ostream_iterator<index_type>{std::cout, ", "});
        std::cout << "|";
        std::copy(begin(_ups), end(_ups),
            std::ostream_iterator<index_type>{std::cout, ", "});
        std::cout << "|";
        std::copy(begin(_downs), end(_downs),
            std::ostream_iterator<index_type>{std::cout, ", "});
        std::cout << "]\n";
    }
#endif

  public:
#if 1
    /// Initialises the "flipper" to a well-defined, but completely unusable
    /// state.
    constexpr RandomFlipperWithMagnetisation() noexcept
        : _ups{}
        , _downs{}
        , _proposed{}
        , _generator{nullptr}
        , _i{}
    {
    }
#endif

    /// \defgroup RandomFlipperWithMagnetisationCopyMove Copy and Move
    /// \{
    RandomFlipperWithMagnetisation(
        RandomFlipperWithMagnetisation const&) = delete;
    RandomFlipperWithMagnetisation(
        RandomFlipperWithMagnetisation&&) noexcept = default;
    RandomFlipperWithMagnetisation& operator       =(
        RandomFlipperWithMagnetisation const&) = delete;
    RandomFlipperWithMagnetisation& operator       =(
        RandomFlipperWithMagnetisation&&) noexcept = default;
    /// \}

  private:
    template <class C>
    auto check_arguments(gsl::span<C> const initial_spin,
        Generator& generator, gsl::span<index_type> workspace,
        index_type const number_flips)
    {
        if (initial_spin.size() == 0) {
            throw_with_trace(std::invalid_argument{
                "Initial spin configuration has size 0."});
        }
        if (number_flips < 0) {
            std::ostringstream msg;
            msg << "Requested number of spin-flips is negative ("
                << number_flips << ").";
            throw_with_trace(std::invalid_argument{msg.str()});
        }
        if (number_flips > initial_spin.size()) {
            std::ostringstream msg;
            msg << "Requested number of spin-flips (" << number_flips
                << ") exceeds the number of spins (" << initial_spin.size()
                << ").";
            throw_with_trace(std::invalid_argument{msg.str()});
        }
        if (number_flips % 2 != 0) {
            std::ostringstream msg;
            msg << "Requested number of spin-flips is odd ("
                << number_flips << ").";
            throw_with_trace(std::invalid_argument{msg.str()});
        }
        Expects(workspace.size()
                == initial_spin.size() + number_flips); // IMPORTANT!
    }

  public:
    /// \brief Creates a new "flipper".
    /// \param number_spins Number of spins in the system. Must not be less than
    ///                     `1`. Proposals will contain indices in the range
    ///                     `[0, ..., number_spins - 1)`.
    /// \param generator Random number generator to use. It must satisfy the
    ///                  UniformRandomBitGenerator concept.
    /// \param number_flips Number of indices in each proposal. Must not be less
    ///                     than `1` and must not exceed \p number_spins.
    template <class C>
    RandomFlipperWithMagnetisation(gsl::span<C> const initial_spin,
        Generator& generator, gsl::span<index_type> workspace,
        index_type const number_flips = 1)
        : _generator{std::addressof(generator)}, _i{0}
    {
        using std::begin, std::end;
        check_arguments(initial_spin, generator, workspace, number_flips);
        Expects(workspace.size()
                == initial_spin.size() + number_flips); // IMPORTANT!

        _proposed = workspace.subspan(0, number_flips);
        // gsl::span{workspace.data(), number_flips};
        auto const begin_ups = begin(workspace) + number_flips;
        std::fill(begin(_proposed), end(_proposed), -1);
        std::iota(begin_ups, end(workspace), 0);
        auto const begin_downs = std::partition(begin_ups, end(workspace),
            [s = initial_spin](auto const i) { return s[i] == C{1}; });
        _ups = workspace.subspan(number_flips, begin_downs - begin_ups);
        // gsl::span{
        // workspace.data() + number_flips, begin_downs - begin_ups};
        _downs = workspace.subspan(
            begin_downs - begin(workspace), end(workspace) - begin_downs);
        // gsl::span{workspace.data() + (begin_downs - begin(workspace)),
        //     end(workspace) - begin_downs};
        if (_ups.size() < _proposed.size() / 2
            || _downs.size() < _proposed.size() / 2) {
            std::ostringstream msg;
            msg << "Given " << _ups.size() << "spins up and "
                << _downs.size()
                << " spins down, it is impossible to perform"
                << number_flips
                << " spin-flips while preserving magnetisation.";
            throw_with_trace(std::invalid_argument{msg.str()});
        }
#if 0
        Ensures((std::all_of(begin(_ups), end(_ups),
            [s = initial_spin](auto const i) { return s[i] == C{1}; })));
        Ensures((std::all_of(begin(_downs), end(_downs),
            [s = initial_spin](auto const i) { return s[i] == C{-1}; })));
        Expects(_ups.size() + _downs.size() == initial_spin.size());
#endif
        shuffle();
        refill_start();
#if 0
        Ensures((std::all_of(begin(_ups), end(_ups),
            [s = initial_spin](auto const i) { return s[i] == C{1}; })));
        Ensures((std::all_of(begin(_downs), end(_downs),
            [s = initial_spin](auto const i) { return s[i] == C{-1}; })));
#endif
    }

  private:
    auto shuffle()
    {
        using std::begin, std::end;
        Expects(_generator != nullptr);
        std::shuffle(begin(_ups), end(_ups), *_generator);
        std::shuffle(begin(_downs), end(_downs), *_generator);
    }

    auto refill_start()
    {
        using std::begin, std::end;
        Expects(_generator != nullptr);
        auto const number_local_flips = _proposed.size() / 2;
        std::copy_n(
            begin(_ups) + _i, number_local_flips, begin(_proposed));
        std::copy_n(begin(_downs) + _i, number_local_flips,
            begin(_proposed) + number_local_flips);
    }

    auto swap_accepted()
    {
        auto const local_proposed_size = _proposed.size() / 2;
        for (auto i = _i; i < _i + local_proposed_size; ++i) {
            std::swap(_ups[i], _downs[i]);
        }
    }

  public:
    auto read() const noexcept(!detail::gsl_can_throw())
        -> gsl::span<index_type const>
    {
        Expects(_generator != nullptr);
        return _proposed;
    }

    auto next(bool const accepted) -> void
    {
        using std::begin, std::end;
        Expects(_generator != nullptr);
        auto const local_proposed_size = _proposed.size() / 2;
        Expects(_i + local_proposed_size <= _ups.size()
                && _i + local_proposed_size <= _downs.size());
        if (accepted) { swap_accepted(); }
        _i += local_proposed_size;
        if (_i + local_proposed_size > _ups.size()
            || _i + local_proposed_size > _downs.size()) {
            _i = 0;
            shuffle();
        }
        refill_start();
        Expects(_i + local_proposed_size <= _ups.size()
                && _i + local_proposed_size <= _downs.size());
    }
};

/// \rst
/// For a given number of spins, an `UniformRandomBitGenerator`_, and a number
/// of spin-flips to propose, generates an infinite `ForwardRange`_ of
/// "proposals". Each proposal is a ``gsl::span<int const>`` containing indices
/// of spins that one should consider flipping.
/// \endrst

template <class State, class Generator>
class MetropolisLocal
    : public ranges::view_facade<MetropolisLocal<State, Generator>> {

    friend ranges::range_access;
    using C = typename State::value_type;
    using R = typename C::value_type;
    using index_type = typename State::index_type;

    using flipper_type = std::variant<RandomFlipper<Generator, index_type>,
        RandomFlipperWithMagnetisation<Generator, index_type>>;

    State*                  _state;
    Generator*              _generator;
    flipper_type            _flipper;
    std::vector<index_type> _workspace;

  public:
    MetropolisLocal() = default;

    MetropolisLocal(State& state, Generator& generator,
        index_type const number_flips,
        bool const preserve_magnetisation)
        : _state{std::addressof(state)}
        , _generator{std::addressof(generator)}
        , _workspace{}
    {
        if (number_flips < 0) {
            std::ostringstream msg;
            msg << "Requested number of spin-flips is negative ("
                << number_flips << ").";
            throw_with_trace(std::invalid_argument{msg.str()});
        }

        if (preserve_magnetisation) {
            _workspace.resize(_state->size_visible() + number_flips);
            auto flipper = RandomFlipperWithMagnetisation{
                _state->spin(), *_generator, {_workspace}, number_flips};
            _flipper = std::move(flipper);
        }
        else {
            _workspace.resize(_state->size_visible());
            _flipper = RandomFlipper{
                _state->spin(), *_generator, {_workspace}, number_flips};
        }
    }

    MetropolisLocal(MetropolisLocal const&) = default;
    MetropolisLocal(MetropolisLocal&&)      = default;
    MetropolisLocal& operator=(MetropolisLocal const&) = default;
    MetropolisLocal& operator=(MetropolisLocal&&) = default;

  private:
    struct cursor {
      private:
        MetropolisLocal* _range;

      public:
        constexpr cursor() noexcept = default;
        constexpr cursor(MetropolisLocal& range) noexcept
            : _range{std::addressof(range)}
        {
        }

        constexpr auto const& read() const noexcept(!detail::gsl_can_throw())
        {
            Expects(_range != nullptr);
            return *_range->_state;
        }

        constexpr auto equal(ranges::default_sentinel /*unused*/) const
        {
            return false;
        }

        auto next() -> void
        {
            Expects(_range != nullptr);
            auto const u =
                std::generate_canonical<R, std::numeric_limits<R>::digits>(
                    *_range->_generator);
            auto const flips = std::visit(
                [](auto const& x) { return x.read(); }, _range->_flipper);
            auto const [log_quot_wf, cache] =
                _range->_state->log_quot_wf(flips);
            // auto const spin = _range->_state->spin();
            auto const probability =
                std::min(R{1.0}, std::norm(std::exp(log_quot_wf)));
            if (u <= probability) {
                _range->_state->update(flips, cache);
                std::visit(
                    [](auto& x) { x.next(true); }, _range->_flipper);
            }
            else {
                std::visit(
                    [](auto& x) { x.next(false); }, _range->_flipper);
            }
        }
    };

    constexpr auto begin_cursor() noexcept(!detail::gsl_can_throw())
        -> cursor
    {
        Expects(_state != nullptr);
        return {*this};
    }
};



namespace detail {

template <class State, class Steps, class Accumulator, class Generator>
decltype(auto) local_metropolis_loop(State& state, Steps&& steps,
    Accumulator&& acc, typename State::index_type const number_flips,
    bool const preserve_magnetisation, Generator& gen)
{
    auto const [start, step, end] = steps;
    MetropolisLocal stream{
        state, gen, number_flips, preserve_magnetisation};
    for (auto const& s : std::move(stream) | ranges::view::take(end)
                             | ranges::view::drop(start)
                             | ranges::view::stride(step)) {
        std::invoke(acc, s);
    }
    return std::forward<Accumulator>(acc);
}

template <class State, class Steps, class Accumulator, class Generator>
decltype(auto) local_metropolis_with_restarts(State& state, Steps&& steps,
    Accumulator&& acc, typename State::index_type const number_flips,
    bool const preserve_magnetisation, Generator& gen,
    typename State::index_type const max_restarts)
{
    using vector_type = typename State::vector_type;
    auto restarts = max_restarts;
    try {
        local_metropolis_loop(
            state, steps, acc, number_flips, preserve_magnetisation, gen);
        return std::forward<Accumulator>(acc);
    }
    catch (use_different_spin<vector_type>& e) {
        std::cerr << e.what() << '\n';
        if (--restarts >= 0) {
            state.spin(std::move(e).get_spin());
            acc.reset();

            std::cerr << "Restarting with s = [";
            auto const spin = state.spin();
            using C = typename State::value_type;
            std::copy(std::begin(spin), std::end(spin),
                std::ostream_iterator<C>{std::cerr, ", "});
            std::cerr << "]\n";

            return local_metropolis_with_restarts(state,
                std::forward<Steps>(steps), std::forward<Accumulator>(acc),
                number_flips, preserve_magnetisation, gen, restarts);
        }
        else {
            return std::forward<Accumulator>(acc);
        }
    }
}

} // namespace detail

template <class Rbm, class Steps, class Accumulator,
    class Generator = std::decay_t<decltype(thread_local_generator())>>
decltype(auto) sequential_local_metropolis(Rbm const& rbm, Steps&& steps,
    Accumulator&& acc, typename Rbm::index_type number_flips = 1,
    std::optional<typename Rbm::index_type> magnetisation = std::nullopt,
    typename Rbm::index_type                max_restarts  = 0,
    Generator&                              gen = thread_local_generator())
{
    using vector_type = typename Rbm::vector_type;
    using C           = typename Rbm::value_type;
    auto initial_spin =
        magnetisation.has_value()
            ? make_random_spin<vector_type>(
                  rbm.size_visible(), magnetisation.value(), gen)
            : make_random_spin<vector_type>(rbm.size_visible(), gen);
    McmcBase<C> state{rbm, std::move(initial_spin)};
    if (magnetisation.has_value()) {
        // TODO: I'm not sure whether this check actually belongs here.
        if (state.size_visible() == std::abs(magnetisation.value())) {
            std::invoke(acc, state);
            return std::forward<Accumulator>(acc);
        }
    }
    return detail::local_metropolis_with_restarts(state,
        std::forward<Steps>(steps), std::forward<Accumulator>(acc),
        number_flips, magnetisation.has_value(), gen, max_restarts);
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_METROPOLIS_LOCAL_HPP
