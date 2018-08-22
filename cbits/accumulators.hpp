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

#ifndef TCM_SWARM_ACCUMULATORS_HPP
#define TCM_SWARM_ACCUMULATORS_HPP

#include "detail/axpy.hpp"
#include "detail/errors.hpp"
#include "detail/scale.hpp"
#include "gradients.hpp"
#include "mcmc_state.hpp"
#include "memory.hpp"
#include "spin_utils.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <utility>

#include <gsl/gsl>

TCM_SWARM_BEGIN_NAMESPACE


/// \brief A counting accumulator.
///
/// This accumulator is used to maintain the current iteration number in the
/// Monte-Carlo loop.
struct Counter {
    using index_type = std::ptrdiff_t;

    constexpr Counter() noexcept : _n{0} {}
    constexpr explicit Counter(index_type const count) noexcept(!detail::gsl_can_throw())
        : _n{count}
    {
        Expects(count >= 0);
    }

    constexpr Counter(Counter const&) noexcept = default;
    constexpr Counter(Counter&&) noexcept      = default;
    constexpr Counter& operator=(Counter const&) noexcept = default;
    constexpr Counter& operator=(Counter&&) noexcept = default;

    constexpr auto operator()() noexcept { ++_n; }

    template <class T>
    constexpr auto operator()(T&& /*unused*/) noexcept
    {
        (*this)();
    }

    /// \brief Resets the counter to 0.
    constexpr auto reset() noexcept -> void { _n = 0; }

    /// \brief Returns the current count.
    constexpr auto count() const noexcept -> index_type { return _n; }

    /// \brief Combines the result with another counter.
    ///
    /// Total count after executing merge is just the sum of two counts.
    constexpr decltype(auto) merge(Counter const other) noexcept(
        !detail::gsl_can_throw())
    {
        constexpr auto max_count = std::numeric_limits<index_type>::max();
        Expects(max_count - _n >= other._n);
        _n += other._n;
        return *this;
    }

  private:
    index_type _n;
};


/// \brief Accumulates the first N statistical moments.
template <std::size_t N, class T, class = void>
struct MomentsAccumulator;

template <std::size_t N, class T>
struct MomentsAccumulator<N, T, std::enable_if_t<std::is_floating_point_v<T>>>
    : public Counter {

    using value_type = T;
    using Counter::count;
    using Counter::index_type;

  private:
    using base = Counter;
    std::array<T, N> _data;

    static constexpr auto _bin(std::size_t const k, std::size_t const n)
        -> std::size_t
    {
        if (k > n) { throw std::invalid_argument{"k should not exceed n."}; }
        if (k == 0 || n == k) { return 1u; }
        return _bin(k - 1, n - 1) + _bin(k, n - 1);
    }

    template <std::size_t K, std::size_t P>
    static constexpr auto bin() noexcept -> std::size_t
    {
        static_assert(0 < K && K <= P);
        constexpr auto c = _bin(K, P);
        return c;
    }

    /// Evaluates a polynomial using Horner's method.
    ///
    /// Suppose that ``cs... = cN, cN-1,..., c0``. Then this function
    /// returns ``c0 + c1 * x + c2 * x^2 + ... + cN * x^N``.
    template <class... Rs>
    static constexpr auto horner(value_type const x, Rs const... cs) noexcept
        -> value_type
    {
        value_type sum{0};
        (static_cast<void>(sum = cs + x * sum), ...);
        return sum;
    }

    constexpr decltype(auto) _mu() noexcept { return _data[0]; }
    constexpr decltype(auto) _mu() const noexcept { return _data[0]; }

    template <std::size_t P>
    constexpr decltype(auto) _M() noexcept
    {
        static_assert(1u < P && P <= N);
        return _data[P - 1];
    }

    template <std::size_t P>
    constexpr decltype(auto) _M() const noexcept
    {
        static_assert(1u < P && P <= N);
        return _data[P - 1];
    }

    template <std::size_t P>
    auto _pth_order_part(value_type const d, std::size_t const n) noexcept
        -> value_type
    {
        static_assert(P > 2u,
            "This function is meant for high-order momenta and will "
            "only work for P larger than 2.");
        constexpr auto sign =
            ((P - 1u) % 2u == 0u) ? value_type{1} : value_type{-1};
        auto const n_min_1 = gsl::narrow_cast<value_type>(n - 1);
        return std::pow(n_min_1 * d, value_type{P})
               * (value_type{1}
                     - sign
                           * std::pow(
                                 value_type{1} / n_min_1, value_type{P - 1}));
    }

    //                       P-2, P-3, ..., 1
    template <std::size_t P, std::size_t... Ks>
    auto update_impl(value_type const d) noexcept -> value_type
    {
        static_assert(P > 2u,
            "This function is meant for high-order momenta and will "
            "only work for P larger than 2.");
        static_assert(std::min({Ks...}) == 1,
            "This function requires `Ks... == P-2, P-3, ..., 1`");
        static_assert(std::max({Ks...}) == P - 2,
            "This function requires `Ks... == P-2, P-3, ..., 1`");
        static_assert(sizeof...(Ks) == P - 2,
            "This function requires `Ks... == P-2, P-3, ..., 1`");
        return _M<P>() + horner(-d, bin<Ks, P>() * _M<P - Ks>()..., 0)
               + _pth_order_part<P>(d, gsl::narrow_cast<std::size_t>(count()));
    }

    //                       0, 1, 2, ..., P-3
    template <std::size_t P, std::size_t... Is>
    auto update(value_type const d,
        std::index_sequence<Is...> /*unused*/) noexcept -> value_type
    {
        static_assert(P > 2u,
            "This function is meant for high-order momenta and will "
            "only work for P larger than 2.");
        static_assert(std::min({Is...}) == 0);
        static_assert(std::max({Is...}) == P - 3);
        static_assert(sizeof...(Is) == P - 2);
        auto const x = update_impl<P, (sizeof...(Is) - Is)...>(d);
        return x;
    }

    template <std::size_t P>
    auto update(value_type const d) noexcept -> value_type
    {
        static_assert(P > 2u,
            "This function is meant for high-order momenta and will "
            "only work for P larger than 2.");
        return update<P>(d, std::make_index_sequence<P - 2u>{});
    }

    //        0, 1, 2, ..., N-3
    template <std::size_t... Ps>
    auto _call(value_type const d,
        std::index_sequence<Ps...> /*unused*/) noexcept -> void
    {
        static_assert(N > 2,
            "This function is meant for high-order momenta and will "
            "only work for N larger than 2.");
        static_assert(std::min({Ps...}) == 0 && std::max({Ps...}) == N - 3
                          && sizeof...(Ps) == N - 2,
            "This function requires `Ps... == 0, 1, 2, ..., N-3");
        std::array<value_type, N - 2> next_gen_Ms = {update<Ps + 3>(d)...};
        std::copy(std::begin(next_gen_Ms), std::end(next_gen_Ms),
            std::begin(_data) + 2u);
    }

  public:
    /// \brief Initialises the accumulator.
    MomentsAccumulator() noexcept : base{}
    {
        std::fill(std::begin(_data), std::end(_data), 0);
    }

    MomentsAccumulator(MomentsAccumulator const&)     = default;
    MomentsAccumulator(MomentsAccumulator&&) noexcept = default;
    MomentsAccumulator& operator=(MomentsAccumulator const&) = default;
    MomentsAccumulator& operator=(MomentsAccumulator&&) noexcept = default;

    /// \brief Records the next sample.
    template <class U,
        class = std::enable_if_t<std::is_convertible_v<U&&, value_type>>>
    auto operator()(U&& next_sample) noexcept -> void
    {
        static_cast<base&>(*this)(); // updates count
        value_type const y{std::forward<U>(next_sample)};
        auto const       delta = y - _mu();
        if (count() == 1) { _mu() = y; }
        else {
            auto const d = delta / gsl::narrow_cast<value_type>(count());
            // Clang-tidy doesn't work well with if constexpr yet
            // NOLINTNEXTLINE
            if constexpr (N > 2) {
                _call(d, std::make_index_sequence<N - 2>{});
            }
            _mu() += d;
            _M<2>() += delta * (y - _mu());
        }
    }

    /// \brief Returns the P'th statistical moment.
    ///
    /// Assumes that count() > 0.
    template <std::size_t P>
    constexpr auto get() const noexcept(!detail::gsl_can_throw()) -> T
    {
        static_assert(1u <= P && P <= N);
        if constexpr (P == 1u) { return _mu(); }
        else {
            Expects(count() > 0);
            return _M<P>() / count();
        }
    }

    /// \brief Resets the accumulator
    auto reset() noexcept -> void
    {
        static_cast<base&>(*this).reset();
        std::fill(std::begin(_data), std::end(_data), 0);
    }

    /// \brief Merges the results with another counter.
    constexpr decltype(auto) merge(MomentsAccumulator const& other)
    {
        if (count() == 0 && other.count() == 0) { return *this; }

        value_type const sum =
            _mu() * count() + other._mu() * other.count();
        static_cast<base&>(*this).merge(other);
        _mu() = sum / gsl::narrow_cast<value_type>(count());

        if constexpr (N >= 2) {
// Clang says that omp pragmas are not allowed in constexpr functions
// #pragma omp simd
            for (std::size_t i = 1; i < N; ++i) {
                // Yes I know what I'm doing: i _is_ within bounds
                // NOLINTNEXTLINE
                _data[i] += other._data[i];
            }
        }

        return *this;
    }

  private:
    template <class U, std::size_t... Is>
    constexpr auto copy_to(
        gsl::span<U, gsl::narrow_cast<std::ptrdiff_t>(N)> const out,
        std::index_sequence<
            Is...> /*unused*/) noexcept(!detail::gsl_can_throw())
    {
        (static_cast<void>(out[Is] = get<Is + 1>()), ...);
    }

  public:
    template <class U,
        class = std::enable_if_t<std::is_convertible_v<value_type, U>>>
    constexpr auto copy_to(
        gsl::span<U, gsl::narrow_cast<std::ptrdiff_t>(N)> const
            out) noexcept(!detail::gsl_can_throw()) -> void
    {
        copy_to(out, std::make_index_sequence<N>{});
    }
};


template <class T>
struct MomentsAccumulator<1, T, std::enable_if_t<std::is_floating_point_v<T>>>
    : Counter {

    static_assert(std::is_trivial_v<T>,
        "Non-trivial types are not (yet) supported.");
    using Counter::index_type;
    using value_type = T;

    using Counter::count;
  private:
    using base = Counter;

  public:
    constexpr MomentsAccumulator() noexcept : base{}, _mu{0} {}
    MomentsAccumulator(MomentsAccumulator const&)     = default;
    MomentsAccumulator(MomentsAccumulator&&) noexcept = default;
    MomentsAccumulator& operator=(MomentsAccumulator const&) = default;
    MomentsAccumulator& operator=(MomentsAccumulator&&) noexcept = default;

    template <class U, class = std::enable_if_t<std::is_convertible_v<U, T>>>
    constexpr auto operator()(U&& x)
    {
        static_cast<base&>(*this)(); // update count
        value_type const y{std::forward<U>(x)};
        _mu += (y - _mu) / gsl::narrow_cast<value_type>(count());
    }

    constexpr auto reset() -> void
    {
        static_cast<base*>(this)->reset();
        _mu = 0;
    }

    constexpr auto mean() noexcept(!detail::gsl_can_throw()) -> value_type
    {
        Expects(count() > 0);
        return _mu;
    }

    // Assumes _n > 0.
    template <std::size_t P>
    constexpr auto get() const noexcept -> T
    {
        static_assert(P == 1u);
        return _mu;
    }

    constexpr decltype(auto) merge(MomentsAccumulator const& other)
    {
        if (count() == 0 && other.count() == 0) { return *this; }
        value_type const sum =
            _mu * count() + other._mu * other.count();
        static_cast<base&>(*this).merge(static_cast<base const&>(other));
        _mu = sum / gsl::narrow_cast<value_type>(count());
        return *this;
    }

    template <class U,
        class = std::enable_if_t<std::is_convertible_v<value_type, U>>>
    constexpr auto copy_to(gsl::span<U, 1> const out) noexcept(
        !detail::gsl_can_throw())
    {
        out[0] = _mu;
    }

  private:
    value_type _mu;
};

template <std::size_t N, class T>
struct MomentsAccumulator<N, std::complex<T>,
    std::enable_if_t<std::is_floating_point_v<T>>> {

    using value_type = std::complex<T>;
    using index_type = typename MomentsAccumulator<N, T>::index_type;

  private:
    // TODO(twesterhout): This is sub-optimal, because both MomentsAccumulators
    // derive from Counter and we end up counting twice... But I think it
    // doesn't matter much in this case.
    MomentsAccumulator<N, T> _real;
    MomentsAccumulator<N, T> _imag;

  public:
    MomentsAccumulator() = default;
    MomentsAccumulator(MomentsAccumulator const&) = default;
    MomentsAccumulator(MomentsAccumulator&&) noexcept = default;
    MomentsAccumulator& operator=(MomentsAccumulator const&) = default;
    MomentsAccumulator& operator=(MomentsAccumulator&&) noexcept = default;

    template <class U,
        class = std::enable_if_t<std::is_convertible_v<U, value_type>>>
    constexpr auto operator()(U&& x) noexcept
    {
        value_type const y{std::forward<U>(x)};
        _real(y.real());
        _imag(y.imag());
    }

    constexpr auto count() const noexcept { return _real.count(); }

    constexpr auto reset() noexcept -> void
    {
        _real.reset();
        _imag.reset();
    }

    template <std::size_t P>
    constexpr auto get() const noexcept(!detail::gsl_can_throw()) -> value_type
    {
        Expects(count() > 0);
        return {_real.template get<P>(), _imag.template get<P>()};
    }

    constexpr decltype(auto) merge(MomentsAccumulator const& other) noexcept
    {
        _real.merge(other._real);
        _imag.merge(other._imag);
        return *this;
    }

  private:
    template <class U, std::size_t... Is>
    constexpr auto copy_to(
        gsl::span<U, gsl::narrow_cast<std::ptrdiff_t>(N)> const out,
        std::index_sequence<
            Is...> /*unused*/) noexcept(!detail::gsl_can_throw())
    {
        (static_cast<void>(out[Is] = get<Is + 1>()), ...);
    }

  public:
    template <class U,
        class = std::enable_if_t<std::is_convertible_v<value_type, U>>>
    constexpr auto copy_to(
        gsl::span<U, gsl::narrow_cast<std::ptrdiff_t>(N)> const
            out) noexcept(!detail::gsl_can_throw())
    {
        copy_to(out, std::make_index_sequence<N>{});
    }
};

template struct MomentsAccumulator<1, float>;
template struct MomentsAccumulator<1, double>;
template struct MomentsAccumulator<1, long double>;
template struct MomentsAccumulator<1, std::complex<float>>;
template struct MomentsAccumulator<1, std::complex<double>>;
template struct MomentsAccumulator<1, std::complex<long double>>;
template struct MomentsAccumulator<2, float>;
template struct MomentsAccumulator<2, double>;
template struct MomentsAccumulator<2, long double>;
template struct MomentsAccumulator<2, std::complex<float>>;
template struct MomentsAccumulator<2, std::complex<double>>;
template struct MomentsAccumulator<2, std::complex<long double>>;
template struct MomentsAccumulator<3, float>;
template struct MomentsAccumulator<3, double>;
template struct MomentsAccumulator<3, long double>;
template struct MomentsAccumulator<3, std::complex<float>>;
template struct MomentsAccumulator<3, std::complex<long double>>;
template struct MomentsAccumulator<4, float>;
template struct MomentsAccumulator<4, double>;
template struct MomentsAccumulator<4, long double>;
template struct MomentsAccumulator<4, std::complex<float>>;
template struct MomentsAccumulator<4, std::complex<long double>>;

template <std::size_t N, class C, class Hamiltonian>
class EnergyAccumulator : public MomentsAccumulator<N, C> {

    using R    = typename C::value_type;
    using base = MomentsAccumulator<N, C>;

  public:
    explicit constexpr EnergyAccumulator(Hamiltonian hamiltonian) noexcept(
        std::is_nothrow_move_constructible_v<Hamiltonian>)
        : base{}, _hamiltonian{std::move(hamiltonian)}
    {
    }

    EnergyAccumulator(EnergyAccumulator const&)     = default;
    EnergyAccumulator(EnergyAccumulator&&) noexcept = default;
    EnergyAccumulator& operator=(EnergyAccumulator const&) = default;
    EnergyAccumulator& operator=(EnergyAccumulator&&) noexcept = default;

    constexpr auto operator()(McmcState const& state) -> void
    {
        static_cast<base&> (*this)(_hamiltonian(state));
    }

  private:
    Hamiltonian _hamiltonian;
};

template class EnergyAccumulator<1, std::complex<float>, std::function<std::complex<float>(McmcState const&)>>;
template class EnergyAccumulator<1, std::complex<double>, std::function<std::complex<float>(McmcState const&)>>;
template class EnergyAccumulator<2, std::complex<float>, std::function<std::complex<float>(McmcState const&)>>;
template class EnergyAccumulator<3, std::complex<float>, std::function<std::complex<float>(McmcState const&)>>;
template class EnergyAccumulator<4, std::complex<float>, std::function<std::complex<float>(McmcState const&)>>;

template <std::size_t N, class C, class Hamiltonian>
class CachingEnergyAccumulator : public MomentsAccumulator<N, C> {

    using base = MomentsAccumulator<N, C>;

  public:
    using typename base::index_type;

    struct Cache {
        C          energy;
        index_type count;
    };

    explicit CachingEnergyAccumulator(Hamiltonian hamiltonian)
        : base{}, _cache{}, _hamiltonian{std::move(hamiltonian)}
    {
    }

    CachingEnergyAccumulator(Hamiltonian hamiltonian, index_type const guess)
        : base{}, _cache{}, _hamiltonian{std::move(hamiltonian)}
    {
        Expects(guess >= 0);
        _cache.rehash(gsl::narrow_cast<std::size_t>(guess));
    }

    // clang-format off
    CachingEnergyAccumulator(CachingEnergyAccumulator const&) = default;
    CachingEnergyAccumulator(CachingEnergyAccumulator&&) noexcept = default;
    CachingEnergyAccumulator& operator=(CachingEnergyAccumulator const&) = default;
    CachingEnergyAccumulator& operator=(CachingEnergyAccumulator&&) noexcept = default;
    // clang-format on

  private:
    auto&       parent() & noexcept { return static_cast<base&>(*this); }

  public:
    auto operator()(McmcState const& state) -> void
    {
        auto spin = to_bitset(state.spin());
        if (auto i = _cache.find(spin); i != std::end(_cache)) {
            auto& cache = i->second;
            parent()(cache.energy);
            ++cache.count;
        }
        else {
            auto const local_energy = _hamiltonian(state);
            parent()(local_energy);
            auto const [iterator, success] =
                _cache.emplace(std::move(spin), Cache{local_energy, 1});
            Ensures(success);
        }
    }

    constexpr auto reset() -> void
    {
        parent().reset();
        for (auto& [_, c] : _cache) {
            c.count = 0;
        }
    }

    constexpr auto const& cache() const& noexcept { return _cache; }
    constexpr auto&       cache() & noexcept { return _cache; }
    constexpr auto        cache() && { return std::move(_cache); }

  private:
    std::unordered_map<std::vector<bool>, Cache> _cache;
    // TODO(twesterhout): This could probably benefit from EBO.
    Hamiltonian _hamiltonian;
};

template class CachingEnergyAccumulator<1, std::complex<float>, std::function<std::complex<float>(McmcState const&)>>;
template class CachingEnergyAccumulator<1, std::complex<double>, std::function<std::complex<float>(McmcState const&)>>;
template class CachingEnergyAccumulator<2, std::complex<float>, std::function<std::complex<float>(McmcState const&)>>;
template class CachingEnergyAccumulator<3, std::complex<float>, std::function<std::complex<float>(McmcState const&)>>;
template class CachingEnergyAccumulator<4, std::complex<float>, std::function<std::complex<float>(McmcState const&)>>;

template <std::size_t N, class C, class Hamiltonian>
class GradientAccumulator : public MomentsAccumulator<N, C> {

    using base = MomentsAccumulator<N, C>;

  public:
    using energies_type  = gsl::span<C>;
    using gradients_type = Gradients<C>;
    using base::count;

    GradientAccumulator(Hamiltonian hamiltonian,
        energies_type const         energy_storage,
        gradients_type const        gradient_storage)
        : _hamiltonian{std::move(hamiltonian)}
        , _energies{energy_storage}
        , _gradients{gradient_storage}
    {
        Expects(_energies.size() == _gradients.template extent<0>());
    }

    auto operator()(McmcState const& state) -> void
    {
        if (count() >= _energies.size()) {
            std::ostringstream msg;
            msg << "GradientAccumulator::operator(): count = " << count()
                << ", but _energies.size() = " << _energies.size() << '\n';
            std::cerr << msg.str() << std::flush;
        }
        Expects(count() < _energies.size());
        Expects(_gradients.template extent<1>() == state.size());

        auto const local_energy = _hamiltonian(state);
        _energies[count()]      = local_energy;
        state.der_log_wf(_gradients[count()]);
        static_cast<base&>(*this)(local_energy);
    }

    auto reset() -> void { static_cast<base*>(this)->reset(); }

  private:
    Hamiltonian    _hamiltonian;
    energies_type  _energies;
    gradients_type _gradients;
};

template <std::size_t N, class C, class Hamiltonian>
class CachingGradientAccumulator : public MomentsAccumulator<N, C> {

    using base = MomentsAccumulator<N, C>;

  public:
    using energies_type  = gsl::span<C>;
    using gradients_type = Gradients<C>;
    using base::count;
    using typename base::index_type;

    class Cache {
        C          _energy; ///< Local energy
        index_type _count;  ///< Number of times this spin
                            /// configuration has been encountered.
        index_type
            _iteration; ///< Iteration at which this spin configuration has
                        /// already been encountered (i.e. iteration where
                        /// to steal the gradient from).

      public:
        constexpr Cache(C const energy) noexcept
            : _energy{energy}, _count{1}, _iteration{-1}
        {
        }

        constexpr Cache(C const energy, index_type const iteration) noexcept(
            !detail::gsl_can_throw())
            : _energy{energy}, _count{1}, _iteration{iteration}
        {
            Expects(_iteration >= 0);
        }

        constexpr Cache(Cache const&) noexcept = default;
        constexpr Cache(Cache&&) noexcept      = default;
        constexpr Cache& operator=(Cache const&) noexcept = default;
        constexpr Cache& operator=(Cache&&) noexcept = default;

        constexpr auto energy() const noexcept { return _energy; }
        constexpr auto count() const noexcept { return _count; }

        constexpr auto iteration() const noexcept(!detail::gsl_can_throw())
        {
            Expects(_count > 0 && _iteration >= 0);
            return _iteration;
        }

        constexpr decltype(auto) record() noexcept
        {
            ++_count;
            return *this;
        }

        constexpr decltype(auto) record(index_type const i) noexcept(
            !detail::gsl_can_throw())
        {
            Expects(i >= 0);
            ++_count;
            _iteration = i;
            return *this;
        }

        constexpr decltype(auto) reset() noexcept
        {
            _count     = 0;
            _iteration = -1;
        }

        constexpr auto gradient_is_known() const noexcept -> bool
        {
            return _count > 0 && _iteration >= 0;
        }
    };


    // clang-format off
    CachingGradientAccumulator(Hamiltonian hamiltonian,
        energies_type const energy_storage, gradients_type const gradient_storage)
        : base{}
        , _energies{energy_storage}
        , _gradients{gradient_storage}
        , _cache{}
        , _hamiltonian{std::move(hamiltonian)}
    // clang-format on
    {
    }

    template <class State>
    auto operator()(State&& state) -> void
    {
        using std::begin, std::end;
        Expects(count() < _energies.size());
        Expects(_gradients.template extent<1>() == state.size());

        auto       spin              = to_bitset(state.spin());
        auto const number_parameters = state.size();
        if (auto c = _cache.find(spin); c != std::end(_cache)) {
            auto& cache = c->second;
            if (cache.gradient_is_known()) {
                auto const current_gradient = _gradients[count()];
                auto const known_gradient   = _gradients[cache.iteration()];
                std::copy(begin(known_gradient), end(known_gradient),
                    begin(current_gradient));
                cache.record();
            }
            else {
                state.der_log_wf(_gradients[count()]);
                cache.record(count());
            }
            _energies[count()] = cache.energy();
            static_cast<base&> (*this)(cache.energy());
        }
        else {
            auto const local_energy = _hamiltonian(state);
            state.der_log_wf(_gradients[count()]);
            _energies[count()]             = local_energy;
            auto const [iterator, success] = _cache.emplace(
                std::move(spin), Cache{local_energy, this->count()});
            Expects(success);
            static_cast<base&> (*this)(local_energy);
        }
    }

    auto reset()
    {
        for (auto& [spin, c] : _cache) {
            c.reset();
        }
        static_cast<base*>(this)->reset();
    }

    constexpr auto const& cache() const& noexcept { return _cache; }
    constexpr auto&       cache() & noexcept { return _cache; }
    constexpr auto        cache() && { return std::move(_cache); }

  private:
    energies_type                                _energies;
    gradients_type                               _gradients;
    std::unordered_map<std::vector<bool>, Cache> _cache;
    Hamiltonian                                  _hamiltonian;
};

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_ACCUMULATORS_HPP

