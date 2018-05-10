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

#ifndef TCM_SWARM_RBM_SPIN_HPP
#define TCM_SWARM_RBM_SPIN_HPP

#include <complex>
#include <memory>
#include <vector>

#include <gsl/gsl>
#include <Vc/Vc>

#include "detail/config.hpp"
#include "detail/debug.hpp"
#include "detail/mkl.hpp"

#include "detail/simd.hpp"
// #include "detail/axpby.hpp"
// #include "detail/copy.hpp"
#include "detail/dotu.hpp"
#include "detail/gemv.hpp"
// #include "detail/lncosh.hpp"
#include "detail/mkl_allocator.hpp"
// #include "detail/scale.hpp"
// #include "detail/random.hpp"


TCM_SWARM_BEGIN_NAMESPACE


namespace detail {

/// \brief Determines which alignment to use for internal vectors (biases and
/// weights).
template <class T, class Abi = Vc::simd_abi::native<T>>
constexpr auto alignment() noexcept -> std::size_t
{
    // Intel MKL suggests to use buffers aligned to at least 64 bytes for
    // optimal performance.
    return std::max<std::size_t>(
        Vc::memory_alignment_v<Vc::simd<T, Abi>>, 64u);
}

struct is_valid_spin_fn {
    template <class R,
        class = std::enable_if_t<std::is_floating_point_v<R>>>
    constexpr auto operator()(R const x) const noexcept -> bool
    {
        return x == R{-1} || x == R{1};
    }

    template <class R,
        class = std::enable_if_t<std::is_floating_point_v<R>>>
    constexpr auto operator()(std::complex<R> const x) const noexcept
        -> bool
    {
        return this->operator()(x.real()) && x.imag() == R{0};
    }

    template <class R>
    auto operator()(gsl::span<R> const x) const
        -> bool
    {
        using std::begin, std::end;
        return std::all_of(begin(x), end(x),
            [this](auto const s) -> bool { return this->operator()(s); });
    }
};

} // namespace detail


/// \verbatim embed:rst:leading-slashes
///   .. note::
///
///      Nice :cpp:class:`tcm::RbmBase`.
///
///   .. note::
///
///      This class currently uses aligned allocation and deallocation routines
///      provided by Intel MKL library. If you want to see a more cross platform
///      or more customisable solution, please, submit a request `here
///      <https://github.com/twesterhout/tcm-swarm/issues>`_
///
/// \endverbatim
template <class T,
    class Allocator = mkl::mkl_allocator<T, detail::alignment<T>()>,
    class           = void>
class RbmBase;

template <class T,
    class Allocator = mkl::mkl_allocator<T, detail::alignment<T>()>,
    class           = void>
struct McmcBase;

template <class R, class Allocator>
class RbmBase<std::complex<R>, Allocator,
    std::enable_if_t<std::is_floating_point_v<R>>> {

  private:
    using C = std::complex<R>;

  public:

    using allocator_type  = Allocator;
    using value_type      = C;
    using vector_type     = std::vector<value_type, allocator_type>;
    using index_type      = typename gsl::span<C>::index_type;
    using size_type       = typename vector_type::size_type;
    using difference_type = typename vector_type::difference_type;

  private:
    vector_type _weights; ///< Weights of the RBM.
    vector_type _visible; ///< Visible bias.
    vector_type _hidden;  ///< Hidden bias.

  public:
    /// \rst
    /// Constructs a new machine with ``size_visible`` visible units and
    /// ``size_hidden`` hidden units. Weights and biases contain unspecified
    /// values.
    ///
    /// .. warning::
    ///
    ///    This function will throw an exception if a negative number of visible
    ///    or hidden spins is passed, or if memory allocation fails.
    ///
    /// \endrst
    RbmBase(index_type const size_visible, index_type const size_hidden)
    {
        if (size_visible < 0) {
            throw std::domain_error{
                "Can't construct tcm::RbmBase with a negative number ("
                + std::to_string(size_visible) + ") of visible units."};
        }
        if (size_hidden < 0) {
            throw std::domain_error{
                "Can't construct tcm::RbmBase with a negative number ("
                + std::to_string(size_hidden) + ") of hidden units."};
        }
        auto const n = gsl::narrow<size_type>(size_visible);
        auto const m = gsl::narrow<size_type>(size_hidden);
        _weights.resize(n * m);
        _visible.resize(n);
        _hidden.resize(m);
    }

    /// \defgroup RbmSpinCopyMove Copy and Move
    /// \{

    RbmBase(RbmBase const&)     = default;
    RbmBase(RbmBase&&) noexcept = default;
    RbmBase& operator=(RbmBase const&) = default;
    RbmBase& operator=(RbmBase&&) noexcept = default;

    /// \}


    /// \brief Returns memory layout (column-major or row-major) of the weight matrix.
    static constexpr auto layout_weights() noexcept
    {
        return mkl::Layout::ColMajor;
    }

    /// \brief Returns number of visible units.
    constexpr auto size_visible() const noexcept -> index_type
    {
        return gsl::narrow_cast<index_type>(_visible.size());
    }

    /// \brief Returns number of hidden units.
    constexpr auto size_hidden() const noexcept -> index_type
    {
        return gsl::narrow_cast<index_type>(_hidden.size());
    }

    /// \brief Returns number of elements in the weight matrix.
    ///
    /// This is equivalent to ``size_visible() * size_hidden()``.
    constexpr auto size_weights() const noexcept -> index_type
    {
        return gsl::narrow_cast<index_type>(_weights.size());
    }

    /// \brief Returns the total number of parameters in the machine.
    constexpr auto size() const noexcept -> index_type
    {
        return size_visible() + size_hidden() + size_weights();
    }

  private:
    constexpr auto ldim_weights() const noexcept(!detail::gsl_can_throw())
        -> index_type
    {
        if constexpr (layout_weights() == mkl::Layout::ColMajor) {
            return size_hidden();
        }
        else {
            return size_visible();
        }
    }

    template <class T>
    TCM_SWARM_FORCEINLINE static constexpr auto matrix_access(T const* m,
        index_type const ldim, index_type const r,
        index_type const c) noexcept(!detail::gsl_can_throw()) -> T
    {
        if constexpr (layout_weights() == mkl::Layout::ColMajor) {
            return m[r + ldim * c];
        }
        else {
            return m[ldim * r + c];
        }
    }

  public:
    /// \defgroup RbmSpinAccessors Getters and Setters
    /// \{

    /// \brief Returns the weights matrix as flattened array.
    constexpr auto weights() const& noexcept -> gsl::span<C const>
    {
        return {_weights};
    }

    /// \overload
    constexpr auto weights() & noexcept -> gsl::span<C>
    {
        return {_weights};
    }

    /// \brief Returns \f$w_{rc}\f$.
    constexpr auto weights(index_type const r, index_type const c) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        Expects(0 <= r && r < size_hidden());
        Expects(0 <= c && c < size_visible());
        return matrix_access(_weights.data(), ldim_weights(), r, c);
    }

    /// \brief Returns visible bias.
    constexpr auto visible() const& noexcept -> gsl::span<C const>
    {
        return {_visible};
    }

    /// \overload
    constexpr auto visible() & noexcept -> gsl::span<C>
    {
        return {_visible};
    }

    /// \brief Returns \f$a_i\f$
    constexpr auto visible(index_type const i) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        Expects(0 <= i && i < size_visible());
        return _visible[i];
    }

    /// \brief Returns
    /// \f[
    ///     \left[ \operatorname{Re}[a_i], \operatorname{Im}[a_i],
    ///            \operatorname{Re}[a_{i+1}], \operatorname{Im}[a_{i+1}],
    ///            \dots,
    ///            \operatorname{Re}[a_{i+K}], \operatorname{Im}[a_{i+K}]
    ///     \right]
    /// \f]
    /// where `K = Vc::simd<R, Abi> / 2`.
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto visible_v(index_type const i,
        Vc::flags::element_aligned_tag flag) const
        -> Vc::simd<R, Abi>
    {
        static_assert(Vc::simd<R, Abi>::size() % 2 == 0);
        Expects((i + Vc::simd<R, Abi>::size() / 2 < size_visible()));
        auto const* p = _visible.data() + i;
        return {static_cast<R const*>(p), flag};
    }

    /// \overload
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto visible_v(index_type const i,
        Vc::flags::vector_aligned_tag flag) const
        -> Vc::simd<R, Abi>
    {
        static_assert(Vc::simd<R, Abi>::size() % 2 == 0);
        Expects((i + Vc::simd<R, Abi>::size() / 2 < size_visible()));
        auto const* p = _visible.data() + i;
        Expects((static_cast<std::intptr_t>(p)
                    % Vc::memory_alignment_v<Vc::simd<R, Abi>> == 0));
        return {static_cast<R const*>(p), flag};
    }

    /// \brief Returns hidden bias.
    constexpr auto hidden() const& noexcept -> gsl::span<C const>
    {
        return {_hidden};
    }

    /// \overload
    constexpr auto hidden() & noexcept -> gsl::span<C>
    {
        return {_hidden};
    }

    /// \brief Returns \f$b_i\f$
    constexpr auto hidden(index_type const i) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        Expects(0 <= i && i < size_hidden());
        return _hidden[i];
    }

    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto hidden_v(index_type const i,
        Vc::flags::element_aligned_tag flag) const
        -> Vc::simd<R, Abi>
    {
        static_assert(Vc::simd<R, Abi>::size() % 2 == 0);
        Expects((i + Vc::simd<R, Abi>::size() / 2 < size_hidden()));
        auto const* p = _hidden.data() + i;
        return {static_cast<R const*>(p), flag};
    }

    /// \overload
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto hidden_v(index_type const i,
        Vc::flags::vector_aligned_tag flag) const
        -> Vc::simd<R, Abi>
    {
        static_assert(Vc::simd<R, Abi>::size() % 2 == 0);
        Expects((i + Vc::simd<R, Abi>::size() / 2 < size_hidden()));
        auto const* p = _hidden.data() + i;
        Expects((static_cast<std::intptr_t>(p)
                    % Vc::memory_alignment_v<Vc::simd<R, Abi>> == 0));
        return {static_cast<R const*>(p), flag};
    }

    /// \}

    /// \verbatim embed:rst:leading-slashes
    /// Calculates :math:`\theta = w\sigma + b`, where :math`w` are the weights
    /// and :math:`b` -- bias of the hidden layer. The result is stored into out.
    ///
    /// .. note::
    ///
    ///    Use this function sparingly, as it is :math:`\mathcal{O}(NM)` in
    ///    complexity.
    ///
    /// \endverbatim
    TCM_SWARM_NOINLINE
    auto theta(gsl::span<C const> const spin, gsl::span<C> const out) const
        noexcept(!detail::gsl_can_throw()) -> void
    {
        using std::begin, std::end;
        Expects(size_hidden() == out.size());
        Expects(detail::is_valid_spin_fn{}(spin));
        auto const b = hidden();
        auto const w = weights();
        // theta := b
        std::copy(begin(b), end(b), begin(out));
        // theta := 1.0 * w * spin + 1.0 * theta
        mkl::gemv(layout_weights(), mkl::Transpose::None, C{1},
            w, spin, C{1}, out);
    }

    auto theta(gsl::span<C const> const spin) const -> vector_type
    {
        using std::begin, std::end;
        Expects(detail::is_valid_spin_fn{}(spin));
        vector_type out(gsl::narrow<size_type>(size_hidden()));
        theta(spin, {out});
        return out;
    }

    /// \brief Calculates \f$\log\psi(\sigma; \mathcal{W})\f$.
    ///
    /// This function is implemented as
    /// \f[
    ///     \log\psi(\sigma; \mathcal{W}) = \operatorname{Re}[a]\cdot\sigma
    ///         + \operatorname{Im}[a]\cdot\sigma + \sum_i\cosh(\theta) \;.
    /// \f]
    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto log_wf(gsl::span<C const> spin, C const sum_logcosh_theta) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        static C const log_of_2 = std::log(C{2});
        Expects(spin.size() == size_visible());
        return gsl::narrow<R>(size_hidden()) * log_of_2
               + mkl::dotu(visible(), spin) + sum_logcosh_theta;
    }

    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto log_wf(gsl::span<C const> spin) const -> C
    {
        static C const log_of_2 = std::log(C{2});
        Expects(spin.size() == size_visible());
        auto const theta = this->theta(spin);
        auto const sum_logcosh_theta =
            sum_log_cosh(gsl::span<C const>{theta});
        auto const x = gsl::narrow<R>(size_hidden()) * log_of_2
                       + mkl::dotu(visible(), spin) + sum_logcosh_theta;
        return x;
    }
};

/// \brief This class caches a few extra things which allow for faster
/// construction of Markov chains, but are not really part of the Rbm.
template <class R, class Allocator>
struct McmcBase<std::complex<R>, Allocator,
    std::enable_if_t<std::is_floating_point_v<R>>> {

  private:
    using C   = std::complex<R>;
    using Rbm = tcm::RbmBase<C, Allocator>;

  public:
    using value_type      = typename Rbm::value_type;
    using index_type      = typename Rbm::index_type;
    using allocator_type  = typename Rbm::allocator_type;
    using vector_type     = typename Rbm::vector_type;
    using size_type       = typename Rbm::size_type;
    using difference_type = typename Rbm::difference_type;

  private:
    Rbm const&  _rbm;
    vector_type _spin;  ///< Current spin configuration \f$\sigma\f$.
    vector_type _theta; ///< Cached \f$\theta\f$ (i.e. \f$b + w \sigma\f$).
    C _sum_logcosh_theta; ///< Cached \f$\sum_i\log\cosh(\theta_i)\f$.
    C _log_psi;           ///< Cached \f$\log\Psi_\mathcal{W}(\sigma)\f$.

  public:
    /// \rst
    /// Initialises the state with given RBM and initial spin configuration.
    /// Size of ``spin`` must match the number of visible units in ``rbm``.
    ///
    /// .. note::
    ///
    ///    This function involves a memory allocation for :math:`\theta` and may
    ///    thus throw.
    /// \endrst
    explicit McmcBase(Rbm const& rbm, vector_type&& spin)
        : _rbm{rbm}, _spin{std::move(spin)}
    {
        TRACE();
        Expects(gsl::narrow<index_type>(_spin.size()) == _rbm.size_visible());
        Expects(detail::is_valid_spin_fn{}(gsl::span<C const>{_spin}));
        _theta             = _rbm.theta(_spin);
        _sum_logcosh_theta = sum_log_cosh(gsl::span<C const>{_theta});
        _log_psi           = _rbm.log_wf(_spin, _sum_logcosh_theta);
    }

    /// \overload
    explicit McmcBase(Rbm const& rbm, gsl::span<R const> const spin)
        : McmcBase{rbm, vector_type{std::begin(spin), std::end(spin)}}
    {
    }

    constexpr auto size_visible() const noexcept { return _rbm.size_visible(); }

    constexpr auto size_hidden() const noexcept { return _rbm.size_hidden(); }

    constexpr auto theta() const& noexcept -> gsl::span<C const>
    {
        return {_theta};
    }

    constexpr auto theta() & noexcept -> gsl::span<C> { return {_theta}; }

    constexpr auto spin() const& noexcept -> gsl::span<C const>
    {
        return {_spin};
    }

    constexpr auto spin() & noexcept -> gsl::span<C> { return {_spin}; }

  private:
    auto flips_are_within_bounds(
        gsl::span<index_type const> const flips) const noexcept -> bool
    {
        using std::begin, std::end;
        return std::all_of(
            begin(flips), end(flips), [n = size_visible()](auto const f) {
                return 0 <= f && f < n;
            });
    }

    auto flips_are_unique(gsl::span<index_type const> const flips) const -> bool
    {
        using std::begin, std::end;
        std::vector<index_type> temp_flips{begin(flips), end(flips)};
        std::sort(begin(temp_flips), end(temp_flips));
        return std::adjacent_find(begin(temp_flips), end(temp_flips))
               == end(temp_flips);
    }

  public:
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
    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto new_theta(
        index_type const i, gsl::span<index_type const> const flips) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        Expects(flips_are_within_bounds(flips));
        C delta{0};
        for (auto const flip : flips) {
            delta += _rbm.weights(i, flip) * _spin[flip];
        }
        return _theta[i] - C{2} * delta;
    }

    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto sum_logcosh_new_theta(gsl::span<index_type const> const flips) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        C sum{0};
        for (size_type i = 0, n = size_hidden(); i < n; ++i) {
            sum += _log_cosh(new_theta(i, flips));
        }
        return sum;
    }

    /// \rst
    /// Given a sequence of spin-flips ``flips``, calculates
    /// :math:`\log\frac{\Psi_\mathcal{W}(\sigma')}{\Psi_\mathcal{W}(\sigma)}`
    /// where :math:`\sigma` is the current spin configuration, and
    /// :math:`\sigma'` is obtained from :math:`\sigma` by flipping spins at
    /// indices indicated by ``flips``.
    ///
    /// .. warning::
    ///
    ///    This function assumes that ``flips`` contains no duplicates.
    ///
    /// Returns a tuple of two elements:
    /// 1) Logarithm of the quotient of the wave functions.
    /// 2) :math:`\operatorname{Tr}[\log\cosh(w\sigma' + b)]`. This is a
    ///    by-product of the calculation and, if passed, to a subsequent call to
    ///    :cpp:func:`update` can greatly speed it up.
    ///
    /// \endrst
    TCM_SWARM_NOINLINE
    TCM_SWARM_PURE
    auto log_quot_wf(gsl::span<index_type const> const flips) const
        noexcept(!detail::gsl_can_throw()) -> std::tuple<C, C>
    {
        Expects(flips_are_within_bounds(flips));
        Expects(flips_are_unique(flips));
#if 0
        auto const log_wf_old = _log_psi; // _rbm.log_wf(_spin);
        vector_type new_spin{_spin};
        for (auto flip : flips) {
            new_spin[gsl::narrow<size_type>(flip)] *= C{-1};
        }
        auto const log_wf_new = _rbm.log_wf(new_spin);
#endif
        auto const sum_logcosh_new = sum_logcosh_new_theta(flips);
        C          delta{0};
        for (auto flip : flips) {
            delta += _rbm.visible(flip) * _spin[flip];
        }
        auto const log_quot_wf =
            sum_logcosh_new - _sum_logcosh_theta - C{2} * delta;
        // Expects(std::abs(log_wf_new - log_wf_old - log_quot_wf) < 1.0E-3);
        return {log_quot_wf, sum_logcosh_new};
    }

    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto log_wf() const noexcept -> C { return _log_psi; }

#if 0
    template <class... SizeTypes>
    auto propose(SizeTypes... flips) const noexcept(noexcept(
        std::declval<McmcState const&>().log_quotient_wf(flips...))) -> R
    {
        return std::min(R{1},
            std::pow(std::abs(std::exp(log_quotient_wf(flips...))), R{2}));
    }
#endif

  private:
    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    auto update_theta(gsl::span<index_type const> const flips) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        for (index_type i = 0, n = size_hidden(); i < n; ++i) {
            _theta[i] = new_theta(i, flips);
        }
    }

    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    auto update_spin(gsl::span<index_type const> const flips) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        for (auto const flip : flips) {
            _spin[flip] = -_spin[flip];
        }
        Ensures(detail::is_valid_spin_fn{}(gsl::span<C const>{_spin}));
    }

  public:
    auto update(gsl::span<index_type const> const flips) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        update_theta(flips);
        update_spin(flips);
        _sum_logcosh_theta = sum_log_cosh(gsl::span<C const>{_theta});
        _log_psi           = _rbm.log_wf(_spin, _sum_logcosh_theta);
    }
};


TCM_SWARM_END_NAMESPACE


#endif // TCM_SWARM_RBM_SPIN_HPP
