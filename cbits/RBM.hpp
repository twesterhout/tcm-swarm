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

#ifndef TCM_SWARM_INTERNAL_RBM_ICC
#define TCM_SWARM_INTERNAL_RBM_ICC

#include <complex>
#include <memory>
#include <random>
#include <utility>

#include "detail/config.hpp"
#include "detail/debug.hpp"
#include "detail/mkl.hpp"

#include "detail/axpby.hpp"
#include "detail/copy.hpp"
#include "detail/dotu.hpp"
#include "detail/gemv.hpp"
#include "detail/lncosh.hpp"
#include "detail/mkl_allocator.hpp"
#include "detail/scale.hpp"

TCM_SWARM_BEGIN_NAMESPACE

template <class T>
struct _Storage_Base {
  private:
    using allocator_type = ::tcm::mkl::mkl_allocator<T>;

  public:
    using alloc_traits    = std::allocator_traits<allocator_type>;
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = T const*;
    using size_type       = mkl::size_type;
    using difference_type = mkl::difference_type;

    struct deleter_type {
        auto operator()(typename alloc_traits::pointer const p) const
            noexcept -> void
        {
            allocator_type alloc{};
            alloc_traits::deallocate(
                alloc, p, typename alloc_traits::size_type{1});
        }
    };

    using array_type = std::unique_ptr<T[], deleter_type>;
    static_assert(sizeof(array_type) == sizeof(pointer),
        "std::unique_ptr with mkl_allocator is incompatible with a "
        "raw C-pointer.");

    static auto _allocate(typename alloc_traits::size_type const n)
    {
        allocator_type alloc{};
        return alloc_traits::allocate(alloc, n);
    }
};

namespace detail {
inline auto pretty_print(std::FILE* stream, float const x)
{
    std::fprintf(stream, "%.1e", static_cast<double>(x));
}

inline auto pretty_print(std::FILE* stream, double const x)
{
    std::fprintf(stream, "%.1e", x);
}

template <class T>
inline auto pretty_print(std::FILE* stream, std::complex<T> const x)
{
    pretty_print(stream, x.real());
    std::fprintf(stream, " + ");
    pretty_print(stream, x.imag());
    std::fprintf(stream, "i");
}

template <class T, class SizeType>
inline auto pretty_print(
    std::FILE* stream, T const* x, SizeType const n)
{
    std::fprintf(stream, "[");
    if (n != 0) { pretty_print(stream, x[0]); }
    for (SizeType i = 1; i < n; ++i) {
        std::fprintf(stream, ", ");
        pretty_print(stream, x[i]);
    }
    std::fprintf(stream, "]");
}
} // namespace detail

template <class T>
struct RbmState : private _Storage_Base<T> {

    //                    n = _size_visible
    //                       +----------+  +  +
    //                       ||         |  |  |
    //                       ||         |  |  |
    //                       ||         |  |  | _visible
    //                       |v         |  |  |
    //                       |          |  |  +
    //     m = k * n         |          |  |
    //       = _size_hidden  | _weights |  | _hidden
    //                       |          |  |
    //                       |          |  |
    //                       |          |  |
    //                       +----------+  +
    //

  public:
    using typename _Storage_Base<T>::alloc_traits;
    using typename _Storage_Base<T>::value_type;
    using typename _Storage_Base<T>::pointer;
    using typename _Storage_Base<T>::const_pointer;
    using typename _Storage_Base<T>::array_type;
    using typename _Storage_Base<T>::size_type;
    using typename _Storage_Base<T>::difference_type;

    array_type _weights;
    array_type _visible;
    array_type _hidden;
    size_type  _size_visible;
    size_type  _size_hidden;

  private:
    static constexpr inline difference_type one = 1;

  public:
    explicit RbmState(
        size_type const size_visible, size_type const size_hidden)
        : _weights{this->_allocate(size_visible * size_hidden)}
        , _visible{this->_allocate(size_visible)}
        , _hidden{this->_allocate(size_hidden)}
        , _size_visible{size_visible}
        , _size_hidden{size_hidden}
    {
    }

    explicit RbmState(RbmState const& other)
        : _weights{this->_allocate(other.size_weights())}
        , _visible{this->_allocate(other.size_visible())}
        , _hidden{this->_allocate(other.size_hidden())}
        , _size_visible{other.size_visible()}
        , _size_hidden{other.size_hidden()}
    {
        unsafe_update_weights(other.weights());
        unsafe_update_visible(other.visible());
        unsafe_update_hidden(other.hidden());
    }

    explicit RbmState(RbmState&& other) noexcept
        : _weights{std::move(other._weights)}
        , _visible{std::move(other._visible)}
        , _hidden{std::move(other._hidden)}
        , _size_visible{other.size_visible()}
        , _size_hidden{other.size_hidden()}
    {
    }

    RbmState& operator=(RbmState const&) = delete;
    RbmState& operator=(RbmState&&) = delete;

    TCM_SWARM_FORCEINLINE
    static constexpr auto layout_weights() noexcept -> mkl::Layout
    {
        return mkl::Layout::ColMajor;
    }

    TCM_SWARM_FORCEINLINE
    constexpr auto ldim_weights() const noexcept -> difference_type
    {
        if constexpr (layout_weights() == mkl::Layout::ColMajor) {
            return static_cast<difference_type>(this->_size_hidden);
        }
        else /*RowMajor layout*/ {
            return static_cast<difference_type>(this->_size_visible);
        }
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto
    size_visible() const noexcept -> size_type
    {
        return _size_visible;
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto
    size_hidden() const noexcept -> size_type
    {
        return _size_hidden;
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto
    size_weights() const noexcept -> size_type
    {
        return size_visible() * size_hidden();
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto visible() const
        & noexcept -> const_pointer
    {
        return _visible.get();
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto visible()
        & noexcept -> pointer
    {
        return _visible.get();
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto hidden() const
        & noexcept -> const_pointer
    {
        return _hidden.get();
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto hidden()
        & noexcept -> pointer
    {
        return _hidden.get();
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto weights() const
        & noexcept -> const_pointer
    {
        return _weights.get();
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto weights()
        & noexcept -> pointer
    {
        return _weights.get();
    }

    [[nodiscard]] TCM_SWARM_FORCEINLINE constexpr auto weight(
        size_type const r, size_type const c) const noexcept -> T
    {
        if constexpr (layout_weights() == mkl::Layout::ColMajor) {
            return _weights[r
                            + static_cast<size_type>(ldim_weights())
                                  * c];
        }
        else /*RowMajor layout*/ {
            return _weights[static_cast<size_type>(ldim_weights()) * r
                            + c];
        }
    }

    TCM_SWARM_FORCEINLINE
    decltype(auto) unsafe_update_weights(
        const_pointer const weights) noexcept
    {
        mkl::copy(
            size_visible() * size_hidden(), weights, this->weights());
        return *this;
    }

    TCM_SWARM_FORCEINLINE
    decltype(auto) unsafe_update_visible(
        const_pointer const visible) noexcept
    {
        mkl::copy(size_visible(), visible, this->visible());
        return *this;
    }

    TCM_SWARM_FORCEINLINE
    decltype(auto) unsafe_update_hidden(
        const_pointer const hidden) noexcept
    {
        mkl::copy(size_hidden(), hidden, this->hidden());
        return *this;
    }

  private:
    auto _pretty_print_weights(std::FILE* stream) const
    {
        fprintf(stream, "w: [");
        if (_size_hidden != 0) {
            for (size_type j = 0; j < _size_visible; ++j) {
                detail::pretty_print(stream, weight(0, j));
                std::fprintf(stream, ", ");
            }
        }
        for (size_type i = 1; i < _size_hidden; ++i) {
            fprintf(stream, "\n    ");
            for (size_type j = 0; j < _size_visible; ++j) {
                detail::pretty_print(stream, weight(i, j));
                std::fprintf(stream, ", ");
            }
        }
        fprintf(stream, "\n   ]");
    }

  public:
    template <class U>
    friend auto pretty_print(
        std::FILE* stream, RbmState<std::complex<U>> const& rbm)
    {
        std::fprintf(stream, "a: ");
        detail::pretty_print(
            stream, rbm._visible.get(), rbm._size_visible);
        std::fprintf(stream, "\n");

        std::fprintf(stream, "b: ");
        detail::pretty_print(
            stream, rbm._hidden.get(), rbm._size_hidden);
        std::fprintf(stream, "\n");

        rbm._pretty_print_weights(stream);
    }
};

namespace detail {

template <class T, class DiffType>
auto _sum_ln_cosh(std::make_unsigned_t<DiffType> const n,
    T const* const x, DiffType const inc_x) noexcept -> T
{
    T sum{0};
    for (DiffType i = 0; i < static_cast<DiffType>(n); ++i) {
        sum += mkl::lncosh(x[inc_x * i]);
    }
    return sum;
}

} // namespace detail

template <class T>
struct McmcState : _Storage_Base<T> {

    using typename _Storage_Base<T>::alloc_traits;
    using typename _Storage_Base<T>::value_type;
    using typename _Storage_Base<T>::pointer;
    using typename _Storage_Base<T>::const_pointer;
    using typename _Storage_Base<T>::array_type;
    using typename _Storage_Base<T>::size_type;
    using typename _Storage_Base<T>::difference_type;

  public:
    RbmState<T>& _rbm;
    array_type   _spin;
    array_type   _theta;

    static constexpr inline difference_type one = 1;

    auto _compute_theta() noexcept -> void
    {
        // theta := hidden
        mkl::copy(_rbm.size_hidden(), _rbm.hidden(), one,
            _theta.get(), one);
        // theta := 1.0 * weights * spin + 1.0 * theta
        mkl::gemv(_rbm.layout_weights(), mkl::Transpose::None,
            _rbm.size_hidden(), _rbm.size_visible(), T{1},
            _rbm.weights(), _rbm.ldim_weights(), _spin.get(), one,
            T{1}, _theta.get(), one);
    }

  public:
    explicit McmcState(RbmState<T>& rbm, array_type&& spin)
        : _rbm{rbm}
        , _spin{std::move(spin)}
        , _theta{this->_allocate(rbm.size_hidden())}
    {
        _compute_theta();
    }

    explicit McmcState(RbmState<T>& rbm, const_pointer const spin)
        : _rbm{rbm}
        , _spin{this->_allocate(rbm.size_visible())}
        , _theta{this->_allocate(rbm.size_hidden())}
    {
        mkl::copy(_rbm.size_visible(), spin, one, _spin.get(), one);
        _compute_theta();
    }

    [[nodiscard]] auto log_wf() const noexcept -> T
    {
        return mkl::dotu(_rbm.size_visible(), _rbm.visible(), one,
                   _spin.get(), one)
               + detail::_sum_ln_cosh(
                     _rbm.size_hidden(), _theta.get(), one);
    }

    template <class... SizeTypes>
    [[nodiscard]] auto _new_theta(
        size_type const i, SizeTypes... flips) const noexcept -> T
    {
        return _theta[i]
               - T{2}
                     * (T{0} + ...
                           + (_rbm.weight(i, flips) * _spin[flips]));
    }

    template <class... SizeTypes>
    [[nodiscard]] auto log_quotient_wf(SizeTypes... flips) const
        noexcept -> T
    {
        auto log_quotient =
            -T{2}
            * (T{0} + ... + (_rbm._visible[flips] * _spin[flips]));
        for (size_type i = 0; i < _rbm.size_hidden(); ++i) {
            log_quotient += mkl::lncosh(_new_theta(i, flips...))
                            - mkl::lncosh(_theta[i]);
        }
        return log_quotient;
    }

    template <class... SizeTypes>
    [[nodiscard]] auto propose(SizeTypes... flips) const noexcept
    {
        return std::min(mkl::real_of_t<T>{1},
            std::pow(std::abs(std::exp(log_quotient_wf(flips...))),
                mkl::real_of_t<T>{2}));
    }

  private:
    template <class... SizeTypes>
    auto _update_theta(SizeTypes... flips) noexcept -> void
    {
        for (size_type i = 0; i < _rbm._size_hidden; ++i) {
            _theta[i] = _new_theta(i, flips...);
        }
    }

    template <class... SizeTypes>
    auto _update_spin(SizeTypes... flips) noexcept -> void
    {
        ((_spin[flips] *= T{-1}), ...);
    }

  public:
    template <class... SizeTypes>
    auto accept(SizeTypes... flips) noexcept -> void
    {
        _update_theta(flips...);
        _update_spin(flips...);
    }

    template <class U>
    friend auto pretty_print(
        std::FILE* stream, McmcState<std::complex<U>> const& mcmc)
    {
        pretty_print(stream, mcmc._rbm);
        std::fprintf(stream, "\n");

        std::fprintf(stream, "theta: ");
        detail::pretty_print(
            stream, mcmc._theta.get(), mcmc._rbm.size_hidden());
        std::fprintf(stream, "\n");

        std::fprintf(stream, "spin: ");
        detail::pretty_print(
            stream, mcmc._spin.get(), mcmc._rbm.size_visible());
    }
};

namespace detail {

template <class T, class SizeType>
auto spin2num(T const* spin, SizeType const n) noexcept
    -> std::uint64_t
{
    if (n == 0) return 0;

    std::uint64_t x = (spin[0] == T{1});
    for (SizeType i = 1; i < n; ++i) {
        x = (x << 1) + (spin[i] == T{1});
    }
    return x;
}

} // namespace detail

template <class T>
auto heisenberg_1d(McmcState<T> const& mcmc) noexcept -> T
{
    using size_type = typename McmcState<T>::size_type;
    if (mcmc._rbm.size_visible() <= 1u) { return 0; }

    auto const component = [&mcmc](auto const i) noexcept->T
    {
        return (mcmc._spin[i] == mcmc._spin[i + 1])
                   ? T{1}
                   : (T{-1}
                         + T{2}
                               * std::exp(
                                     mcmc.log_quotient_wf(i, i + 1)));
    };

    T sum{0};
    for (size_type i = 0; i < mcmc._rbm.size_visible() - 1; ++i) {
        sum += component(i);
    }

    // std::fprintf(stderr, "%llu\n",
    //     detail::spin2num(mcmc._spin.get(), mcmc._rbm.size_visible()));
    return sum;
}

namespace detail {
namespace {

    template <class T,
        class = std::enable_if_t<std::is_floating_point<T>::value>>
    TCM_SWARM_FORCEINLINE auto _delta_well_update(T const k,
        T const p, T const x, T const r1, T const r2) noexcept -> T
    {
        return p
               + std::copysign(std::log(T{1} / r1), T{2} * r2 - T{1})
                     * std::abs(x - p) / k;
    }

    template <class T,
        class = std::enable_if_t<std::is_floating_point<T>::value>>
    TCM_SWARM_FORCEINLINE auto _delta_well_update(T const k,
        std::complex<T> const p, std::complex<T> const x, T const r1,
        T const r2) noexcept -> std::complex<T>
    {
        return p
               + std::copysign(std::log(T{1} / r1), T{2} * r2 - T{1})
                     * std::complex<T>{std::abs((x - p).real()),
                           std::abs((x - p).imag())}
                     / k;
    }

    template <class T, class SizeType,
        class =
            std::enable_if_t<std::is_floating_point_v<
                                 T> && std::is_unsigned_v<SizeType>>>
    TCM_SWARM_FORCEINLINE auto _delta_well_update(SizeType const n,
        T const k, std::complex<T>* const p,
        std::complex<T> const* const x, T const* const u) noexcept
    {
        for (SizeType i = 0; i < n; ++i) {
            p[i] = _delta_well_update(
                k, p[i], x[i], u[2 * i], u[2 * i + 1]);
        }
    }

} // unnamed namespace
} // namespace detail

template <class Rbm>
auto delta_well_update(
    mkl::real_of_t<typename Rbm::value_type> const k, Rbm& p,
    Rbm const&                                      x,
    mkl::real_of_t<typename Rbm::value_type> const* u) noexcept
{
    detail::_delta_well_update(
        x.size_weights(), k, p.weights(), x.weights(), u);
    std::advance(u, x.size_weights());
    detail::_delta_well_update(
        x.size_visible(), k, p.visible(), x.visible(), u);
    std::advance(u, x.size_visible());
    detail::_delta_well_update(
        x.size_hidden(), k, p.hidden(), x.hidden(), u);
}

template <class T>
auto axpby(T const alpha, RbmState<T> const& x, T const beta,
    RbmState<T>& y) noexcept
{
    using difference_type = typename RbmState<T>::difference_type;
    constexpr difference_type one{1};

    mkl::axpby(x.size_weights(), alpha, x.weights(), one, beta,
        y.weights(), one);
    mkl::axpby(x.size_visible(), alpha, x.visible(), one, beta,
        y.visible(), one);
    mkl::axpby(x.size_hidden(), alpha, x.hidden(), one, beta,
        y.hidden(), one);
}

template <class T>
auto scale(T const alpha, RbmState<T>& x) noexcept
{
    using difference_type = typename RbmState<T>::difference_type;
    constexpr difference_type one{1};

    mkl::scale(x.size_weights(), alpha, x.weights(), one);
    mkl::scale(x.size_visible(), alpha, x.visible(), one);
    mkl::scale(x.size_hidden(), alpha, x.hidden(), one);
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_INTERNAL_RBM_ICC
