
#include "spin_state.hpp"

TCM_SWARM_BEGIN_NAMESPACE

TCM_SWARM_FORCEINLINE
auto SpinState::initialise() -> void
{
    _rbm->theta(_as_c()->spin(), theta());
    _sum_log_cosh_theta = sum_log_cosh(_as_c()->theta());
    _log_psi            = _rbm->log_wf(_as_c()->spin(), _sum_log_cosh_theta);
}

TCM_SWARM_SYMBOL_EXPORT
SpinState::SpinState(Rbm const& rbm, buffer_type&& initial_spin)
    : McmcState{}
    , _rbm{std::addressof(rbm)}
    , _spin{std::move(initial_spin)}
    , _theta{std::get<0>(Rbm::allocate_buffer(rbm.size_hidden()))}
{
    Expects(detail::is_aligned<Rbm::alignment()>(_data(_spin)));
    Expects(is_valid_spin(spin()));
    initialise();
}

SpinState::SpinState(Rbm const& rbm, gsl::span<C const> spin)
    : SpinState{rbm, buffer_from_span(spin)}
{
}

SpinState::~SpinState() noexcept
{
}

TCM_SWARM_FORCEINLINE
constexpr auto SpinState::size_visible() const noexcept -> index_type
{
    return _rbm->size_visible();
}

TCM_SWARM_FORCEINLINE
constexpr auto SpinState::size_hidden() const noexcept -> index_type
{
    return _rbm->size_hidden();
}

TCM_SWARM_FORCEINLINE
constexpr auto SpinState::size_weights() const noexcept -> index_type
{
    return _rbm->size_weights();
}

TCM_SWARM_FORCEINLINE
constexpr auto SpinState::size() const noexcept -> index_type
{
    return _rbm->size();
}

TCM_SWARM_FORCEINLINE
auto SpinState::theta() const& noexcept -> gsl::span<C const>
{
    return {_data(_theta), size_hidden()};
}

TCM_SWARM_FORCEINLINE
auto SpinState::spin() const& noexcept -> gsl::span<C const>
{
    return {_data(_spin), size_visible()};
}

TCM_SWARM_FORCEINLINE
auto SpinState::theta() & noexcept -> gsl::span<C>
{
    return {_data(_theta), size_hidden()};
}

TCM_SWARM_FORCEINLINE
auto SpinState::spin() & noexcept -> gsl::span<C>
{
    return {_data(_spin), size_visible()};
}

template <class Abi>
TCM_SWARM_FORCEINLINE
auto SpinState::theta_v(
    index_type const i, Vc::flags::vector_aligned_tag const flag) const
    noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
{
    using V                    = Vc::simd<R, Abi>;
    constexpr auto vector_size = static_cast<index_type>(V::size());
    static_assert(vector_size % 2 == 0);
    Expects(0 <= i && i < size_hidden());
    Expects((i & (vector_size - 1)) == 0);
    auto const* p = _data(_theta) + i;
    Expects(detail::is_aligned<Vc::memory_alignment_v<V>>(p));
    return copy_from<Abi>(p, flag);
}

auto SpinState::do_spin(gsl::span<C const> const new_spin) noexcept(
    !detail::gsl_can_throw()) -> void
{
    using std::begin, std::end;
    Expects(new_spin.size() == size_visible());
    Expects(is_valid_spin(new_spin));
    std::copy(begin(new_spin), end(new_spin), _data(_spin));
    initialise();
}

auto SpinState::do_spin() const noexcept -> gsl::span<C const>
{
    return spin();
}

/// \rst
/// Given an index :math:`i` and a set ``flips`` of spin-flips, calculates
/// :math:`\theta' = w\sigma' + b`, where :math:`\sigma'` is obtained from
/// the current spin configuration by flipping spins at indices indicated by
/// ``flips``.
///
/// Internally, the following formula is used:
///
/// .. math::
///
///    \theta_i' = \theta_i - 2\sum_{j\in\mathtt{flips}} w_{ij}\sigma_j \;.
///
///
// clang-format off
template <std::ptrdiff_t Extent>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_PURE
auto SpinState::new_theta(index_type const i,
    gsl::span<index_type const, Extent> const flips) const
    // clang-format on
    noexcept(!detail::gsl_can_throw()) -> C
{
    C delta{0};
    if constexpr (Extent >= 0) {
        for (auto const flip : flips) {
            delta += _rbm->weights(i, flip)
                     * _spin[gsl::narrow_cast<std::size_t>(flip)];
        }
    }
    else {
        for (auto const flip : flips) {
            delta += _rbm->weights(i, flip)
                     * _spin[gsl::narrow_cast<std::size_t>(flip)];
        }
    }
    return _theta[i] - C{2} * delta;
}

// clang-format off
template <std::ptrdiff_t Extent, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_PURE
auto SpinState::new_theta_v(index_type const i,
    gsl::span<index_type const, Extent> const flips,
    Vc::flags::vector_aligned_tag const flag) const
// clang-format on
    noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
{
    using V = Vc::simd<R, Abi>;
    std::complex<V> delta{V{0}, V{0}};
    for (auto const flip : flips) {
        delta += _rbm->weights_v(i, flip, Vc::flags::vector_aligned)
                 * V{_spin[gsl::narrow_cast<std::size_t>(flip)].real()};
    }
    return theta_v(i, flag) - V{2} * delta;
}

// clang-format off
template <std::ptrdiff_t Extent>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_PURE
auto SpinState::sum_log_cosh_new_theta(
    gsl::span<index_type const, Extent> const flips) const
        noexcept(!detail::gsl_can_throw()) -> C
// clang-format on
{
    C sum{0};
    for (size_type i = 0, n = size_hidden(); i < n; ++i) {
        sum += _log_cosh(new_theta(i, flips));
    }
    return sum;
}

template <std::ptrdiff_t Extent>
auto SpinState::sum_log_cosh_new_theta_v(
    gsl::span<index_type const, Extent> const flips) const
    noexcept(!detail::gsl_can_throw()) -> C
{
    using V                    = Vc::simd<R, Vc::simd_abi::native<R>>;
    constexpr auto vector_size = static_cast<index_type>(V::size());

    std::complex<V> sum{V{0}, V{0}};
    for (index_type i = 0; i < size_hidden(); i += vector_size) {
        // std::cerr << "i = " << i << '\n' << std::flush;
        sum += _log_cosh(new_theta_v(i, flips, Vc::flags::vector_aligned));
    }

    return std::complex{Vc::reduce(sum.real()), Vc::reduce(sum.imag())};
}

template <std::ptrdiff_t Extent>
TCM_SWARM_FORCEINLINE
auto SpinState::_do_log_quot_wf(
    gsl::span<index_type const, Extent> const flips) const
    noexcept(!detail::gsl_can_throw()) -> std::tuple<C, Cache>
{
    auto const sum_log_cosh_new = sum_log_cosh_new_theta_v(flips);
    // auto const sum_log_cosh_new_2 = sum_log_cosh_new_theta_v(flips);
    // std::cerr << sum_log_cosh_new << " vs. " << sum_log_cosh_new_2 << '\n' << std::flush;
    C          delta{0};
    for (auto flip : flips) {
        delta +=
            _rbm->visible(flip) * _spin[gsl::narrow_cast<std::size_t>(flip)];
    }
    auto const log_quot_wf =
        sum_log_cosh_new - _sum_log_cosh_theta - C{2} * delta;
    return {log_quot_wf, {sum_log_cosh_new}};
}

auto SpinState::do_log_quot_wf(gsl::span<index_type const> const flips) const
    -> std::tuple<C, std::any>
{
    // TODO: Optimisation for the case when flips.size() is small

    auto const [x, cache] = _do_log_quot_wf(flips);
    return {x, {cache}};
}

auto SpinState::do_log_wf() const noexcept -> C { return _log_psi; }

// clang-format off
template <std::ptrdiff_t Extent>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
auto SpinState::update_theta(gsl::span<index_type const, Extent> const flips)
    noexcept(!detail::gsl_can_throw()) -> void
// clang-format on
{
    for (index_type i = 0, n = size_hidden(); i < n; ++i) {
        _theta[i] = new_theta(i, flips);
    }
}

// clang-format off
template <std::ptrdiff_t Extent>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
auto SpinState::update_theta_v(gsl::span<index_type const, Extent> const flips)
    noexcept(!detail::gsl_can_throw()) -> void
// clang-format on
{
    using V                    = Vc::simd<R, Vc::simd_abi::native<R>>;
    constexpr auto vector_size = static_cast<index_type>(V::size());
    auto*          data        = _data(_theta);
    for (index_type i = 0; i < size_hidden(); i += vector_size, data +=
                                                                vector_size) {
        copy_to(new_theta_v(i, flips, Vc::flags::vector_aligned), data,
            Vc::flags::vector_aligned);
    }
}

// clang-format off
template <std::ptrdiff_t Extent>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
auto SpinState::update_spin(gsl::span<index_type const, Extent> const flips)
    noexcept(!detail::gsl_can_throw()) -> void
// clang-format on
{
    for (auto const flip : flips) {
        auto const i = gsl::narrow_cast<std::size_t>(flip);
        _spin[i]     = -_spin[i];
    }
    Ensures(is_valid_spin(_as_c()->spin()));
}

// clang-format off
template <std::ptrdiff_t Extent>
TCM_SWARM_FORCEINLINE
auto SpinState::_do_update(gsl::span<index_type const, Extent> const flips,
    Cache const cache) noexcept(!detail::gsl_can_throw()) -> void
// clang-format on
{
    auto const x = sum_log_cosh_new_theta_v(flips);
    update_theta_v(flips);
    update_spin(flips);
    _sum_log_cosh_theta = cache.sum_log_cosh;
    _log_psi = _rbm->log_wf(spin(), _sum_log_cosh_theta);
}

// clang-format off
template <std::ptrdiff_t Extent>
TCM_SWARM_FORCEINLINE
auto SpinState::_do_update(gsl::span<index_type const, Extent> const flips)
    noexcept(!detail::gsl_can_throw()) -> void
// clang-format on
{
    update_theta_v(flips);
    update_spin(flips);
    _sum_log_cosh_theta = sum_log_cosh(_as_c()->theta());
    _log_psi           = _rbm->log_wf(spin(), _sum_log_cosh_theta);
}

auto SpinState::do_der_log_wf(gsl::span<C> const out) const
    noexcept(!detail::gsl_can_throw()) -> void
{
    _rbm->der_log_wf(spin(), theta(), out);
}

auto SpinState::do_size_visible() const noexcept -> index_type
{
    return size_visible();
}

auto SpinState::do_size() const noexcept -> index_type
{
    return size();
}

auto SpinState::do_update(gsl::span<index_type const> const flips,
    std::any const& cache) -> void
{
    if (cache.has_value()) {
        _do_update(flips, std::any_cast<Cache>(cache));
    }
    else {
        _do_update(flips);
    }
}

TCM_SWARM_END_NAMESPACE
