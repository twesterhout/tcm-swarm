#ifndef TCM_SWARM_SPIN_STATE_HPP
#define TCM_SWARM_SPIN_STATE_HPP

#include <any>
#include <complex>
#include <gsl/pointers>
#include <gsl/span>

#include "mcmc_state.hpp"
#include "rbm_spin_float.hpp"

TCM_SWARM_BEGIN_NAMESPACE

struct SpinState : public McmcState {
  public:
    using McmcState::C;
    using McmcState::index_type;
    using McmcState::R;
    using McmcState::size_type;
    using Rbm         = _tcm_Rbm;
    using buffer_type = Rbm::buffer_type;

  private:
    gsl::not_null<Rbm const*> _rbm;
    buffer_type _spin;     ///< Current spin configuration \f$\sigma\f$.
    buffer_type _theta;    ///< Cached \f$\theta\f$ (i.e. \f$b + w \sigma\f$).
    C _sum_log_cosh_theta; ///< Cached \f$\sum_i\log\cosh(\theta_i)\f$.
    C _log_psi;            ///< Cached \f$\log\Psi_\mathcal{W}(\sigma)\f$.

    struct Cache {
        C sum_log_cosh;
    };

    constexpr auto const* _as_c() const noexcept { return this; }

    static auto*       _data(buffer_type& x) noexcept { return x.get(); }
    static auto const* _data(buffer_type const& x) noexcept { return x.get(); }

    static auto buffer_from_span(gsl::span<C const> x) -> buffer_type
    {
        using std::begin, std::end;
        auto [x_buffer, _ignored_] = Rbm::allocate_buffer(x.size());
        std::copy(begin(x), end(x), _data(x_buffer));
        return std::move(x_buffer);
    }

  public:
    TCM_SWARM_SYMBOL_IMPORT
    SpinState(Rbm const& rbm, buffer_type&& initial_spin);

    SpinState(Rbm const& rbm, gsl::span<C const> spin);

    SpinState(SpinState const&) = delete;
    SpinState(SpinState&&)      = default;
    SpinState& operator=(SpinState const&) = delete;
    SpinState& operator=(SpinState&&) = default;

    auto spin() const & noexcept -> gsl::span<C const>;
    auto spin() &&      = delete;
    auto spin() const&& = delete;

    auto theta() const & noexcept -> gsl::span<C const>;
    auto theta() &&      = delete;
    auto theta() const&& = delete;

    virtual ~SpinState() noexcept override;

  private:
    virtual auto do_size_visible() const noexcept -> index_type override;
    virtual auto do_size() const noexcept -> index_type override;
    virtual auto do_log_wf() const noexcept -> C override;
    virtual auto do_log_quot_wf(gsl::span<index_type const> flips) const
        -> std::tuple<C, std::any> override;
    virtual auto do_der_log_wf(gsl::span<C> const out) const
        noexcept(!detail::gsl_can_throw()) -> void override;
    virtual auto do_update(gsl::span<index_type const> flips,
        std::any const&                                cache) -> void override;
    virtual auto do_spin(gsl::span<C const> spin) noexcept(
        !detail::gsl_can_throw()) -> void override;
    virtual auto do_spin() const noexcept -> gsl::span<C const> override;

    auto initialise() -> void;

    constexpr auto size_visible() const noexcept -> index_type;
    constexpr auto size_hidden() const noexcept -> index_type;
    constexpr auto size_weights() const noexcept -> index_type;
    constexpr auto size() const noexcept -> index_type;

    auto spin() & noexcept -> gsl::span<C>;
    auto theta() & noexcept -> gsl::span<C>;

    decltype(auto) spin(gsl::span<C const> new_spin) noexcept(
        !detail::gsl_can_throw());

    template <class Abi = Vc::simd_abi::native<R>>
    auto theta_v(
        index_type i, Vc::flags::vector_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>;

    template <std::ptrdiff_t Extent>
    auto new_theta(
        index_type i, gsl::span<index_type const, Extent> flips) const
        noexcept(!detail::gsl_can_throw()) -> C;

    template <std::ptrdiff_t Extent, class Abi = Vc::simd_abi::native<R>>
    auto new_theta_v(index_type i, gsl::span<index_type const, Extent> flips,
        Vc::flags::vector_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>;

    template <std::ptrdiff_t Extent>
    auto sum_log_cosh_new_theta(gsl::span<index_type const, Extent> flips) const
        noexcept(!detail::gsl_can_throw()) -> C;

    template <std::ptrdiff_t Extent>
    auto sum_log_cosh_new_theta_v(gsl::span<index_type const, Extent> flips) const
        noexcept(!detail::gsl_can_throw()) -> C;

    template <std::ptrdiff_t Extent>
    auto _do_log_quot_wf(gsl::span<index_type const, Extent> flips) const
        noexcept(!detail::gsl_can_throw()) -> std::tuple<C, Cache>;

    template <std::ptrdiff_t Extent>
    auto update_theta(gsl::span<index_type const, Extent> flips) noexcept(
        !detail::gsl_can_throw()) -> void;

    template <std::ptrdiff_t Extent>
    auto update_theta_v(gsl::span<index_type const, Extent> flips) noexcept(
        !detail::gsl_can_throw()) -> void;

    template <std::ptrdiff_t Extent>
    auto update_spin(gsl::span<index_type const, Extent> flips) noexcept(
        !detail::gsl_can_throw()) -> void;

    template <std::ptrdiff_t Extent>
    auto _do_update(gsl::span<index_type const, Extent> flips,
        Cache cache) noexcept(!detail::gsl_can_throw()) -> void;

    template <std::ptrdiff_t Extent>
    auto _do_update(gsl::span<index_type const, Extent> flips) noexcept(
        !detail::gsl_can_throw()) -> void;
};

TCM_SWARM_END_NAMESPACE

// TODO(twesterhout): This does not belong here, but I'm yet to figure out a
// better way.
template <class Generator>
TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::make_state(std::optional<index_type> const magnetisation,
    Generator& generator) const -> std::unique_ptr<tcm::McmcState>
{
    using std::begin, std::end;
    if (magnetisation.has_value()) {
        auto const m = *magnetisation;
        if (std::abs(m) > size_visible()) {
            std::ostringstream msg;
            msg << size_visible()
                << " spins can't have a total magnetisation of " << m << ".";
            tcm::throw_with_trace(std::domain_error{msg.str()});
        }
        if ((size_visible() + m) % 2 != 0) {
            std::ostringstream msg;
            msg << size_visible()
                << " spins can't have a total magnetisation of " << m
                << ". `size_visible() + magnetisation` must be even.";
            tcm::throw_with_trace(std::domain_error{msg.str()});
        }
        auto const n            = size_visible();
        auto const number_ups   = (n + m) / 2;
        auto const number_downs = (n - m) / 2;
        auto spin_buffer = std::get<0>(allocate_buffer(size_visible()));
        auto spin = gsl::span{spin_buffer.get(), size_visible()};
        std::fill_n(begin(spin), number_ups, C{1});
        std::fill_n(begin(spin) + number_ups, number_downs, C{-1});
        std::shuffle(begin(spin), end(spin), generator);
        Ensures(tcm::detail::magnetisation_fn{}(spin) == m);
        auto state = std::make_unique<tcm::SpinState>(*this, std::move(spin_buffer));
        return std::unique_ptr<tcm::McmcState>(state.release());
    }
    else {
        auto [spin, _ignored_] = allocate_buffer(size_visible());
        std::generate(
            _data(spin), _data(spin) + size_visible(), [&generator]() -> C {
                std::uniform_int_distribution<int> dist{0, 1};
                return {R{2} * gsl::narrow<R>(dist(generator)) - R{1}, 0};
            });
        auto state = std::make_unique<tcm::SpinState>(*this, std::move(spin));
        return std::unique_ptr<tcm::McmcState>(state.release());
    }
}


#endif // TCM_SWARM_SPIN_STATE_HPP
