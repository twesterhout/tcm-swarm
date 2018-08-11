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

#include "detail/config.hpp"
#include "detail/errors.hpp"
#include "spin.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <utility>

#include <gsl/span>
#include <gsl/multi_span>

TCM_SWARM_BEGIN_NAMESPACE

template <class IndexType = long>
struct Counter {
    static_assert(std::is_integral_v<IndexType>,
        "tcm::Counter: `IndexType` must be an integral type.");

    using index_type = IndexType;

    constexpr Counter() noexcept
        : _n{0}
    {
    }

    constexpr Counter(index_type const count)
        : _n{count}
    {
        if (count < 0) {
            throw_with_trace(std::invalid_argument{
                "Can't create a counter with negative count ("
                + std::to_string(count) + ")."});
        }
    }

    constexpr Counter(Counter const&) noexcept = default;
    constexpr Counter(Counter&&) noexcept      = default;
    constexpr Counter& operator=(Counter const&) noexcept = default;
    constexpr Counter& operator=(Counter&&) noexcept = default;

    constexpr auto operator()() noexcept(!detail::gsl_can_throw())
    {
        constexpr auto max_count = std::numeric_limits<index_type>::max();
        Expects(_n < max_count);
        ++_n;
    }

    template <class T>
    constexpr auto operator()(T&& /*unused*/) noexcept(
        !detail::gsl_can_throw())
    {
        operator()();
    }

    constexpr auto reset() noexcept -> void { _n = 0; }

    constexpr auto count() const noexcept -> index_type { return _n; }

    constexpr decltype(auto) merge(Counter const& other) noexcept
    {
        _n += other._n;
        return *this;
    }

  private:
    index_type _n;
};

template <std::size_t N, class T>
struct MomentsAccumulator : public Counter<> {

    static_assert(std::is_trivial_v<T>,
        "Non-trivial types are not (yet) supported.");
    using Counter<>::index_type;
    using value_type = T;

  private:
    using base = Counter<>;

  private:
    std::array<T, N> _data;

  private:

    static constexpr auto _bin(
        std::size_t const k, std::size_t const n) -> std::size_t
    {
        if (k > n) throw std::invalid_argument{"k should not exceed n."};
        if (k == 0 || n == k) return 1u;
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
    static constexpr auto horner(
        value_type const x, Rs const... cs) noexcept -> value_type
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
        auto const n_min_1 = gsl::narrow<value_type>(n - 1);
        return std::pow(n_min_1 * d, P)
               * (value_type{1}
                     - sign * std::pow(value_type{1} / n_min_1, P - 1));
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
               + _pth_order_part<P>(d, this->count());
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
        static_assert(std::min({Ps...}) == 0
                          && std::max({Ps...}) == N - 3
                          && sizeof...(Ps) == N - 2,
            "This function requires `Ps... == 0, 1, 2, ..., N-3");
        std::array<value_type, N - 2> next_gen_Ms = {update<Ps + 3>(d)...};
        std::copy(std::begin(next_gen_Ms), std::end(next_gen_Ms),
                  std::begin(_data) + 2u);
    }

    template <std::size_t... Is>
    constexpr MomentsAccumulator(
        std::index_sequence<Is...> /*unnused*/) noexcept
        : base{}, _data{(static_cast<void>(Is), value_type{0})...}
    {
    }

  public:
    constexpr MomentsAccumulator() noexcept
        : MomentsAccumulator(std::make_index_sequence<N>{})
    {
    }

    template <class U,
        class = std::enable_if_t<std::is_convertible_v<U&&, value_type>>>
    auto operator()(U&& next_sample) -> void
    {
        static_cast<base*>(this)->operator()();
        value_type const          y{std::forward<U>(next_sample)};
        auto const                delta = y - _mu();
        if (this->count() == 1) {
            _mu() = y;
            return;
        }
        auto const d = delta / gsl::narrow<value_type>(this->count());
        if constexpr (N > 2) {
            _call(d, std::make_index_sequence<N - 2>{});
        }
        _mu() += d;
        _M<2>() += delta * (y - _mu());
    }

    // Assumes _n > 0.
    template <std::size_t P>
    constexpr auto get() const noexcept -> T
    {
        static_assert(1u <= P && P <= N);
        if constexpr (P == 1u) {
            return _mu();
        }
        else {
            return _M<P>() / this->count();
        }
    }

    constexpr decltype(auto) merge(MomentsAccumulator const& other)
    {
        if (this->count() == 0 && other.count() == 0) { return *this; }

        value_type const sum =
            _mu() * this->count() + other._mu() * other.count();
        static_cast<base*>(this)->merge(other);
        _mu() = sum / gsl::narrow<value_type>(this->count());

        if constexpr (N >= 2) {
#pragma omp simd
            for (std::size_t i = 1; i < N; ++i) {
                _data[i] += other._data[i];
            }
        }

        return *this;
    }
};

template <class T>
struct MomentsAccumulator<1, T> : Counter<> {

    static_assert(std::is_trivial_v<T>,
        "Non-trivial types are not (yet) supported.");
    using Counter<>::index_type;
    using value_type = T;

  private:
    using base = Counter<>;

  public:
    MomentsAccumulator()                          = default;
    MomentsAccumulator(MomentsAccumulator const&) = default;
    MomentsAccumulator(MomentsAccumulator&&)      = default;
    MomentsAccumulator& operator=(MomentsAccumulator const&) = default;
    MomentsAccumulator& operator=(MomentsAccumulator&&) = default;

    template <class U,
        class = std::enable_if_t<std::is_convertible_v<U, T>>>
    constexpr auto operator()(U&& x) noexcept(!detail::gsl_can_throw())
    {
        static_cast<base*>(this)->operator()();
        if (this->count() == 1) { _mu = value_type{std::forward<U>(x)}; }
        else {
            _mu += (value_type{std::forward<U>(x)} - _mu)
                   / gsl::narrow<value_type>(this->count());
        }
    }

    constexpr auto reset() noexcept
    {
        static_cast<base*>(this)->reset();
        _mu = value_type{};
    }

    constexpr auto mean() noexcept -> value_type
    {
        Expects(this->count() != 0);
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
        if (this->count() == 0 && other.count() == 0) { return *this; }
        value_type const sum =
            _mu * this->count() + other._mu * other.count();
        static_cast<base*>(this)->merge(other);
        _mu = sum / gsl::narrow<value_type>(this->count());
        return *this;
    }

  private:
    value_type _mu;
};

template <std::size_t N, class C, class Hamiltonian>
class EnergyAccumulator {
    using R = typename C::value_type;

    // This is sub-optimal, because both MomentsAccumulators derive from Counter
    // and we end up counting twice... But I think it doesn't matter much in this
    // case.
    std::tuple<MomentsAccumulator<N, R>, MomentsAccumulator<N, R>,
        Hamiltonian>
        _payload;

  protected:
    constexpr auto const& real() const& noexcept
    {
        return std::get<0>(_payload);
    }

    constexpr auto& real() & noexcept { return std::get<0>(_payload); }

    constexpr auto real() && noexcept
    {
        return std::move(std::get<0>(_payload));
    }

    constexpr auto const& imag() const& noexcept
    {
        return std::get<1>(_payload);
    }

    constexpr auto& imag() & noexcept { return std::get<1>(_payload); }

    constexpr auto imag() && noexcept
    {
        return std::move(std::get<1>(_payload));
    }

    constexpr auto const& hamiltonian() const noexcept
    {
        return std::get<2>(_payload);
    }

  public:
    constexpr EnergyAccumulator(Hamiltonian&& hamiltonian)
        : _payload{{}, {}, std::move(hamiltonian)}
    {
    }

    constexpr EnergyAccumulator(Hamiltonian const& hamiltonian)
        : _payload{MomentsAccumulator<N, R>{}, MomentsAccumulator<N, R>{},
              Hamiltonian{hamiltonian}}
    {
    }

    EnergyAccumulator(EnergyAccumulator const&) = default;
    EnergyAccumulator(EnergyAccumulator&&)      = default;
    EnergyAccumulator& operator=(EnergyAccumulator const&) = default;
    EnergyAccumulator& operator=(EnergyAccumulator&&) = default;

    constexpr decltype(auto) operator()(C const local_energy)
    {
        real()(local_energy.real());
        imag()(local_energy.imag());
        return *this;
    }

    template <class State>
    constexpr decltype(auto) operator()(State&& state)
    {
        auto const local_energy = hamiltonian()(std::forward<State>(state));
        return this->operator()(local_energy);
    }

    constexpr auto reset() -> void
    {
        real().reset();
        imag().reset();
    }

    template <std::size_t P>
    constexpr auto get() const noexcept(!detail::gsl_can_throw()) -> C
    {
        static_assert(1u <= P && P <= N);
        return {real().template get<P>(), imag().template get<P>()};
    }

    constexpr auto count() const noexcept { return real().count(); }

    constexpr decltype(auto) merge(EnergyAccumulator const& other)
    {
        real().merge(other.real());
        imag().merge(other.imag());
        return *this;
    }
};


template <std::size_t N, class C, class Hamiltonian>
class CachingEnergyAccumulator : public EnergyAccumulator<N, C, Hamiltonian> {

    struct element_type {
        C   energy;
        int count;
    };

    std::unordered_map<std::vector<bool>, element_type> _cache;

  private:
    constexpr decltype(auto) base() noexcept
    {
        return static_cast<EnergyAccumulator<N, C, Hamiltonian>&>(*this);
    }

    constexpr decltype(auto) base() const noexcept
    {
        return static_cast<EnergyAccumulator<N, C, Hamiltonian> const&>(
            *this);
    }

  public:
    CachingEnergyAccumulator(Hamiltonian&& hamiltonian)
        : EnergyAccumulator<N, C, Hamiltonian>{std::move(hamiltonian)}
        , _cache{}
    {
    }

    CachingEnergyAccumulator(Hamiltonian const& hamiltonian)
        : EnergyAccumulator<N, C, Hamiltonian>{hamiltonian}, _cache{}
    {
    }

    CachingEnergyAccumulator(CachingEnergyAccumulator const&) = default;
    CachingEnergyAccumulator(CachingEnergyAccumulator&&)      = default;
    CachingEnergyAccumulator& operator=(CachingEnergyAccumulator const&) = default;
    CachingEnergyAccumulator& operator=(CachingEnergyAccumulator&&) = default;

    template <class State>
    constexpr decltype(auto) operator()(State&& state)
    {
        auto spin = to_bitset(state.spin());
        C local_energy;
        if (auto i = _cache.find(spin); i != std::end(_cache)) {
            auto& cache  = i->second;
            local_energy = cache.energy;
            ++cache.count;
        }
        else {
            local_energy =
                this->hamiltonian()(std::forward<State>(state));
            auto const [iterator, success] =
                _cache.insert({std::move(spin), {local_energy, 1}});
            Expects(success);
        }
        base()(local_energy);
        return *this;
    }

    constexpr decltype(auto) reset()
    {
        for (auto& value : _cache) {
            value.second.count = 0;
        }
        base().reset();
    }

    constexpr auto const& cache() const& noexcept
    {
        return _cache;
    }

    constexpr auto& cache() & noexcept
    {
        return _cache;
    }

    constexpr auto cache() &&
    {
        return std::move(_cache);
    }
};

template <std::size_t N, class C, class Hamiltonian>
class GradientAccumulator : public EnergyAccumulator<N, C, Hamiltonian> {

    using base = EnergyAccumulator<N, C, Hamiltonian>;

    using energies_type = gsl::span<C, gsl::dynamic_extent>;
    using gradients_type =
        gsl::multi_span<C, gsl::dynamic_extent, gsl::dynamic_extent>;
    // using index_type = typename energies_type::index_type;

    energies_type  _energies;
    gradients_type _gradients;


  public:
    using base::count;
    using base::hamiltonian;

    GradientAccumulator(Hamiltonian ham,
        energies_type const         energy_storage,
        gradients_type const        gradient_storage)
        : base{std::move(ham)}
        , _energies{energy_storage}
        , _gradients{gradient_storage}
    {
        Expects(_energies.size() == _gradients.template extent<0>());
    }

    template <class State>
    decltype(auto) operator()(State& state)
    {
        Expects(count() < _energies.size());
        Expects(_gradients.template extent<1>() == state.size());

        auto const local_energy = hamiltonian()(state);
        _energies[count()]      = local_energy;
        state.der_log_wf(_gradients[count()]);
        static_cast<base*>(this)->operator()(local_energy);
        return *this;
    }

    auto reset()
    {
        static_cast<base*>(this)->reset();
    }
};

template <std::size_t N, class C, class Hamiltonian>
class CachingGradientAccumulator
    : public EnergyAccumulator<N, C, Hamiltonian> {

    class cache_type {
        C   _energy;
        int _count;
        int _iteration;

      public:
        constexpr cache_type(C const energy) noexcept
            : _energy{energy}, _count{1}, _iteration{-1}
        {
        }

        constexpr cache_type(C const energy, int const iteration) noexcept(
            !detail::gsl_can_throw())
            : _energy{energy}, _count{1}, _iteration{iteration}
        {
            Expects(_iteration >= 0);
        }

        constexpr cache_type(cache_type const&) noexcept = default;
        constexpr cache_type(cache_type&&) noexcept      = default;
        constexpr cache_type& operator                   =(
            cache_type const&) noexcept = default;
        constexpr cache_type& operator=(cache_type&&) noexcept = default;

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

        constexpr decltype(auto) record(int const i) noexcept(
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

    using base = EnergyAccumulator<N, C, Hamiltonian>;
    using typename base::index_type;

    gsl::span<C>                                      _energies;
    gsl::span<C>                                      _gradients;
    std::unordered_map<std::vector<bool>, cache_type> _cache;

    auto gradient(
        index_type const i, index_type const number_parameters) const
        noexcept(!detail::gsl_can_throw()) -> gsl::span<C const>
    {
        Expects((i + 1) * number_parameters <= _gradients.size());
        return _gradients.subspan(
            i * number_parameters, number_parameters);
    }

    auto gradient(index_type const i,
        index_type const
            number_parameters) noexcept(!detail::gsl_can_throw())
        -> gsl::span<C>
    {
        Expects((i + 1) * number_parameters <= _gradients.size());
        return _gradients.subspan(
            i * number_parameters, number_parameters);
    }

  public:
    // clang-format off
    CachingGradientAccumulator(Hamiltonian hamiltonian,
        gsl::span<C> const energy_storage, gsl::span<C> const gradient_storage)
        : base{std::move(hamiltonian)}
        , _energies{energy_storage}
        , _gradients{gradient_storage}
        , _cache{}
    // clang-format on
    {
        Expects(_energies.size() == 0 && _gradients.size() == 0
                || (_energies.size() > 0 && _gradients.size() > 0
                       && _gradients.size() % _energies.size() == 0));
    }

    template <class State>
    decltype(auto) operator()(State&& state)
    {
        using std::begin, std::end;
        Expects(this->count() < _energies.size());
        auto       spin              = to_bitset(state.spin());
        auto const number_parameters = state.size();
        auto const current_gradient =
            gradient(this->count(), state.size());
        if (auto c = _cache.find(spin); c != std::end(_cache)) {
            auto& cache = c->second;
            if (cache.gradient_is_known()) {
                auto const known_gradient =
                    gradient(cache.iteration(), state.size());
                std::copy(begin(known_gradient), end(known_gradient),
                    begin(current_gradient));
                cache.record();
            }
            else {
                state.def_log_wf(current_gradient);
                cache.record(this->count());
            }
            _energies[this->count()] = cache.energy();
            static_cast<base*>(this)->operator()(cache.energy());
        }
        else {
            auto const local_energy = this->hamiltonian()(state);
            state.def_log_wf(current_gradient);
            _energies[this->count()]       = local_energy;
            auto const [iterator, success] = _cache.insert(
                {std::move(spin), {local_energy, this->count()}});
            Expects(success);
            static_cast<base*>(this)->operator()(local_energy);
        }
        return *this;
    }

    auto reset()
    {
        for (auto& [spin, c] : _cache) {
            c.reset();
        }
        static_cast<base*>(this)->reset();
    }
};

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_ACCUMULATORS_HPP

