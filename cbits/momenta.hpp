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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
// #include <iomanip>
// #include <iostream>
#include <utility>

#include "detail/config.hpp"

TCM_SWARM_BEGIN_NAMESPACE

template <std::size_t N, class T, class R = T>
struct momenta_accumulator {

  private:
    std::array<R, N> _data;
    std::size_t      _n;

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
    static constexpr auto horner(R const x, Rs const... cs) noexcept
        -> R
    {
        R sum{0};
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
    auto _pth_order_part(R const d, std::size_t const n) noexcept
        -> R
    {
        static_assert(P > 2u,
            "This function is meant for high-order momenta and will "
            "only work for P larger than 2.");
        constexpr auto sign    = ((P - 1u) % 2u == 0u) ? R{1} : R{-1};
        auto const     n_min_1 = static_cast<R>(n - 1);
        return std::pow(n_min_1 * d, P)
               * (R{1} - sign * std::pow(R{1} / n_min_1, P - 1));
    }

    //                       P-2, P-3, ..., 1
    template <std::size_t P, std::size_t... Ks>
    auto update_impl(R const d) noexcept -> R
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
               + _pth_order_part<P>(d, _n);
    }

    //                       0, 1, 2, ..., P-3
    template <std::size_t P, std::size_t... Is>
    auto update(R const d,
        std::index_sequence<Is...> /*unused*/) noexcept -> R
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
    auto update(R const d) noexcept -> R
    {
        static_assert(P > 2u,
            "This function is meant for high-order momenta and will "
            "only work for P larger than 2.");
        return update<P>(d, std::make_index_sequence<P - 2u>{});
    }

    //        0, 1, 2, ..., N-3
    template <std::size_t... Ps>
    auto _call(R const d,
        std::index_sequence<Ps...> /*unused*/) noexcept -> void
    {
        static_assert(N > 2,
            "This function is meant for high-order momenta and will "
            "only work for N larger than 2.");
        static_assert(std::min({Ps...}) == 0
                          && std::max({Ps...}) == N - 3
                          && sizeof...(Ps) == N - 2,
            "This function requires `Ps... == 0, 1, 2, ..., N-3");
        std::array<R, N - 2> next_gen_Ms = {update<Ps + 3>(d)...};
        std::copy(std::begin(next_gen_Ms), std::end(next_gen_Ms),
                  std::begin(_data) + 2u);
    }

    template <std::size_t... Is>
    constexpr momenta_accumulator(
        std::index_sequence<Is...> /*unnused*/) noexcept
        : _data{(static_cast<void>(Is), R{0})...}, _n{0u}
    {
    }

  public:
    constexpr momenta_accumulator() noexcept
        : momenta_accumulator(std::make_index_sequence<N>{})
    {
    }

    auto operator()(T const next_sample) noexcept -> void
    {
        ++_n;

        auto const y     = static_cast<R>(next_sample);
        auto const delta = y - _mu();

        if (_n == 1) {
            _mu() = y;
            return;
        }

        auto const d = delta / static_cast<R>(_n);
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
            return _M<P>() / _n;
        }
    }
};

TCM_SWARM_END_NAMESPACE
