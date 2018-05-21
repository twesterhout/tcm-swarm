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
#include "detail/errors.hpp"
#include "detail/mkl.hpp"

#include "detail/simd.hpp"
// #include "detail/axpby.hpp"
#include "detail/dotu.hpp"
#include "detail/gemv.hpp"
#include "detail/geru.hpp"
#include "detail/tanh.hpp"
// #include "detail/lncosh.hpp"
#include "detail/mkl_allocator.hpp"
#include "detail/scale.hpp"
// #include "detail/random.hpp"

#include "spin.hpp"

#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostics push
#pragma clang diagnostics ignore "-Wzero-as-null-pointer-constant"
#endif

#include <boost/core/demangle.hpp>

#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostics pop
#endif

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

} // namespace detail

#if !defined(TCM_SWARM_NOCHECK_ALIGNMENT)
#define TCM_SWARM_IS_ALIGNED(pointer, alignment)                          \
    (reinterpret_cast<std::intptr_t>(pointer) % alignment == 0)
#else
#define TCM_SWARM_IS_ALIGNED(pointer, alignment) true
#endif

/// \rst
/// Nice :cpp:class:`tcm::RbmBase`.
///
///   .. note::
///
///      This class currently uses aligned allocation and deallocation routines
///      provided by Intel MKL library. If you want to see a more cross platform
///      or more customisable solution, please, submit a request `here
///      <https://github.com/twesterhout/tcm-swarm/issues>`_
///
/// \endrst
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
    using self_type = RbmBase<std::complex<R>, Allocator>;
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
            throw_with_trace(std::invalid_argument{
                "Number of visible units is negative ("
                + std::to_string(size_visible) + ")."});
        }
        if (size_hidden < 0) {
            throw_with_trace(std::invalid_argument{
                "Number of visible units is negative ("
                + std::to_string(size_visible) + ")."});
        }

        auto const n = gsl::narrow<size_type>(size_visible);
        auto const m = gsl::narrow<size_type>(size_hidden);
        _weights.resize(n * m);
        _visible.resize(n);
        _hidden.resize(m);

        Ensures(
            TCM_SWARM_IS_ALIGNED(_weights.data(), detail::alignment<R>()));
        Ensures(
            TCM_SWARM_IS_ALIGNED(_visible.data(), detail::alignment<R>()));
        Ensures(TCM_SWARM_IS_ALIGNED(_hidden.data(), detail::alignment<R>()));
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
    constexpr auto ldim_weights() const noexcept -> index_type
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

    /// \overload
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto weights_v(index_type const r, index_type const c,
        Vc::flags::element_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> Vc::simd<R, Abi>
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        static_assert(vector_size % 2 == 0);
        static_assert(layout_weights() == mkl::Layout::ColMajor);
        Expects(0 <= r && r + vector_size / 2 <= size_hidden());
        Expects(0 <= c && c < size_visible());
        auto const* p = _weights.data() + r + ldim_weights() * c;
        // Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
        return {reinterpret_cast<R const*>(p), flag};
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
#if 0
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
#endif

    /// \overload
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto visible_v(
        index_type const i, Vc::flags::vector_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> Vc::simd<R, Abi>
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        static_assert(vector_size % 2 == 0);
        Expects((i + vector_size / 2 < size_visible()));
        auto const* p = _visible.data() + i;
        Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
        return {reinterpret_cast<R const*>(p), flag};
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

#if 0
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
#endif

    /// \overload
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto hidden_v(
        index_type const i, Vc::flags::vector_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> Vc::simd<R, Abi>
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        static_assert(vector_size % 2 == 0);
        Expects((i + vector_size / 2 < size_hidden()));
        auto const* p = _hidden.data() + i;
        Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
        return {reinterpret_cast<R const*>(p), flag};
    }

    /// \}

    /// \rst
    /// Calculates :math:`\theta = w\sigma + b`, where :math`w` are the weights
    /// and :math:`b` -- bias of the hidden layer. The result is stored into
    /// ``out``.
    ///
    /// .. note::
    ///
    ///    Use this function sparingly, as it is :math:`\mathcal{O}(NM)` in
    ///    complexity.
    ///
    /// \endrst
    TCM_SWARM_NOINLINE
    auto theta(gsl::span<C const> const spin, gsl::span<C> const out) const
        noexcept(!detail::gsl_can_throw()) -> void
    {
        using std::begin, std::end;
        Expects(size_hidden() == out.size());
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));
        auto const b = hidden();
        auto const w = weights();
        // theta := b
        std::copy(begin(b), end(b), begin(out));
        // theta := 1.0 * w * spin + 1.0 * theta
        mkl::gemv(layout_weights(), mkl::Transpose::None, C{1},
            w, spin, C{1}, out);
    }

    /// \overload
    auto theta(gsl::span<C const> const spin) const -> vector_type
    {
        using std::begin, std::end;
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));
        vector_type out(gsl::narrow<size_type>(size_hidden()));
        theta(spin, {out});
        return out;
    }

    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto log_wf(
        gsl::span<C const> const spin, C const sum_logcosh_theta) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        static C const log_of_2 = std::log(C{2});
        Expects(spin.size() == size_visible());
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));
        return gsl::narrow<R>(size_hidden()) * log_of_2
               + mkl::dotu(visible(), spin) + sum_logcosh_theta;
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
    auto log_wf(gsl::span<C const> const spin) const -> C
    {
        static C const log_of_2 = std::log(C{2});
        Expects(spin.size() == size_visible());
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));
        auto const theta = this->theta(spin);
        auto const sum_logcosh_theta =
            sum_log_cosh(gsl::span<C const>{theta});
        auto const x = gsl::narrow<R>(size_hidden()) * log_of_2
                       + mkl::dotu(visible(), spin) + sum_logcosh_theta;
        return x;
    }

    auto der_log_wf(gsl::span<C const> const spin,
        gsl::span<C const> const theta, gsl::span<C> const out) const
    {
        using std::begin, std::end;
        Expects(spin.size() == size_visible());
        Expects(theta.size() == size_hidden());
        Expects(out.size() == size());
        std::copy(begin(spin), end(spin), begin(out));
        mkl::tanh(theta, out.subspan(spin.size(), theta.size()));
        gsl::span<C const> const spin_part = out.subspan(0, spin.size());
        gsl::span<C const> const tanh_part =
            out.subspan(spin.size(), theta.size());
        mkl::geru(C{1.0}, tanh_part, spin_part,
            out.subspan(
                spin.size() + theta.size(), spin.size() * theta.size()),
            layout_weights());
        mkl::scale(R{0.5}, out);
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
    using self_type = McmcBase<std::complex<R>, Allocator>;

  public:
    using value_type      = typename Rbm::value_type;
    using index_type      = typename Rbm::index_type;
    using allocator_type  = typename Rbm::allocator_type;
    using vector_type     = typename Rbm::vector_type;
    using size_type       = typename Rbm::size_type;
    using difference_type = typename Rbm::difference_type;

  private:
    gsl::not_null<Rbm const*> _rbm;
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
    McmcBase(Rbm const& rbm, vector_type&& spin)
        : _rbm{std::addressof(rbm)}, _spin{std::move(spin)}
    {
        if (gsl::narrow<index_type>(_spin.size()) != _rbm->size_visible()) {
            throw_with_trace(std::invalid_argument{
                "Number of spins is not equal to the number of visible "
                "units in the RBM: "
                + std::to_string(_spin.size())
                + " != " + std::to_string(_rbm->size_visible()) + "."});
        }
        Expects(TCM_SWARM_IS_VALID_SPIN(gsl::span<C const>{_spin}));
        _theta             = _rbm->theta(_spin);
        _sum_logcosh_theta = sum_log_cosh(gsl::span<C const>{_theta});
        _log_psi           = _rbm->log_wf(_spin, _sum_logcosh_theta);
    }

    /// \overload
    explicit McmcBase(Rbm const& rbm, gsl::span<R const> const spin)
        : McmcBase{rbm, vector_type{std::begin(spin), std::end(spin)}}
    {
    }

    McmcBase(McmcBase const&) = delete;
    McmcBase(McmcBase&&) noexcept = default;
    McmcBase& operator=(McmcBase const&) = delete;
    McmcBase& operator=(McmcBase &&) = default;

    constexpr auto size_visible() const noexcept
    {
        return _rbm->size_visible();
    }

    constexpr auto size_hidden() const noexcept
    {
        return _rbm->size_hidden();
    }

    constexpr auto size_weights() const noexcept
    {
        return _rbm->size_weights();
    }

    constexpr auto theta() const& noexcept -> gsl::span<C const>
    {
        return {_theta};
    }

    constexpr auto theta() & noexcept -> gsl::span<C> { return {_theta}; }

    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto theta_v(index_type const i,
        Vc::flags::vector_aligned_tag flag) const
        -> Vc::simd<R, Abi>
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        static_assert(vector_size % 2 == 0);
        Expects((i + vector_size / 2 <= size_hidden()));
        auto const* p = _theta.data() + i;
        Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
        return {reinterpret_cast<R const*>(p), flag};
    }

    constexpr auto spin() const& noexcept -> gsl::span<C const>
    {
        return {_spin};
    }

    constexpr auto spin() & noexcept -> gsl::span<C> { return {_spin}; }

    decltype(auto) spin(vector_type&& new_spin) noexcept
    {
        *this = McmcBase{*_rbm, std::move(new_spin)};
        return *this;
    }

  private:
    auto flips_are_within_bounds(
        gsl::span<index_type const> const flips) const noexcept -> bool
    {
#if !defined(TCM_SWARM_NOCHECK_FLIPS_BOUNDS)
        using std::begin, std::end;
        return std::all_of(
            begin(flips), end(flips), [n = size_visible()](auto const f) {
                return 0 <= f && f < n;
            });
#else
        return true;
#endif
    }

    auto flips_are_unique(gsl::span<index_type const> const flips) const
#if !defined(TCM_SWARM_NOCHECK_FLIPS_UNIQUE)
        -> bool
    {
        using std::begin, std::end;
        std::vector<index_type> temp_flips{begin(flips), end(flips)};
        std::sort(begin(temp_flips), end(temp_flips));
        return std::adjacent_find(begin(temp_flips), end(temp_flips))
               == end(temp_flips);
#else
        noexcept -> bool
    {
        return true;
#endif
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
        Expects(flips_are_unique(flips));
        C delta{0};
        for (auto const flip : flips) {
            delta += _rbm->weights(i, flip) * _spin[flip];
        }
        return _theta[i] - C{2} * delta;
    }

    template <class Abi = Vc::simd_abi::native<R>>
    auto new_theta_v(index_type const     i,
        gsl::span<index_type const> const flips,
        Vc::flags::vector_aligned_tag     flag) const
        noexcept(!detail::gsl_can_throw()) -> Vc::simd<R, Abi>
    {
        Expects(flips_are_within_bounds(flips));
        Expects(flips_are_unique(flips));
        Vc::simd<R, Abi> delta = 0;
        for (auto const flip : flips) {
            delta += _rbm->weights_v(i, flip, Vc::flags::element_aligned)
                     * _spin[flip].real();
        }
        return theta_v(i, flag) - R{2} * delta;
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

    template <class Abi = Vc::simd_abi::native<R>>
    TCM_SWARM_NOINLINE auto sum_logcosh_new_theta_v(
        gsl::span<index_type const> const flips) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        auto           rest        = size_hidden() % vector_size;
        V              sum_real    = 0;
        V              sum_imag    = 0;
        C              sum_rest    = 0;
        for (index_type i = 0; i <= size_hidden() - vector_size;
             i += vector_size) {
            auto [a, b] = _deinterleave(
                new_theta_v(i, flips, Vc::flags::vector_aligned),
                new_theta_v(i + vector_size / 2, flips,
                    Vc::flags::vector_aligned));
            std::tie(a, b) = _log_cosh(a, b);
            sum_real += a;
            sum_imag += b;
        }
        for (index_type i = size_hidden() - rest; i < size_hidden(); ++i) {
            sum_rest += _log_cosh(new_theta(i, flips));
        }
        return std::complex{Vc::reduce(sum_real), Vc::reduce(sum_imag)}
               + sum_rest;
    }

  private:

    struct Cache {
        C sum_log_cosh;
    };

  public:

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
        noexcept(!detail::gsl_can_throw()) -> std::tuple<C, Cache>
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
        auto const sum_logcosh_new = sum_logcosh_new_theta_v(flips);
        C delta{0};
        for (auto flip : flips) {
            delta += _rbm->visible(flip) * _spin[flip];
        }
        auto const log_quot_wf =
            sum_logcosh_new - _sum_logcosh_theta - C{2} * delta;
        // Expects(std::abs(log_wf_new - log_wf_old - log_quot_wf) < 1.0E-3);
        return {log_quot_wf, {sum_logcosh_new}};
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

    template <class Abi = Vc::simd_abi::native<R>>
    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    auto update_theta_v(gsl::span<index_type const> const flips) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        auto           rest        = size_hidden() % (vector_size / 2);
        auto*          data        = reinterpret_cast<R*>(_theta.data());
        for (index_type i = 0; i < size_hidden() - rest;
             i += vector_size / 2, data += vector_size) {
            new_theta_v(i, flips, Vc::flags::vector_aligned)
                .copy_to(data, Vc::flags::vector_aligned);
        }
        for (index_type i = size_hidden() - rest; i < size_hidden(); ++i) {
            _theta[i] = new_theta(i, flips);
        }
    }

    template <std::ptrdiff_t Extent>
    auto
    update_spin(gsl::span<index_type const, Extent> const flips) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        for (auto const flip : flips) {
            _spin[flip] = -_spin[flip];
        }
        Ensures(TCM_SWARM_IS_VALID_SPIN(gsl::make_span(_spin)));
    }

  public:
    template <std::ptrdiff_t Extent>
    auto update(gsl::span<index_type const, Extent> const flips,
        Cache const cache) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        auto const x = sum_logcosh_new_theta_v(flips);
        update_theta_v(flips);
        update_spin(flips);
        _sum_logcosh_theta = cache.sum_log_cosh;
        if (_sum_logcosh_theta
               != sum_log_cosh(gsl::span<C const>{_theta})) {
            std::ostringstream msg;
            msg << _sum_logcosh_theta << " vs. "
                << sum_log_cosh(
                       static_cast<self_type const&>(*this).theta())
                << " vs. " << x;
            throw_with_trace(std::runtime_error{msg.str()});
        }
        _log_psi = _rbm->log_wf(_spin, _sum_logcosh_theta);
    }

    template <std::ptrdiff_t Extent>
    auto update(gsl::span<index_type const, Extent> const flips) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        update_theta_v(flips);
        update_spin(flips);
        _sum_logcosh_theta = sum_log_cosh(gsl::span<C const>{_theta});
        _log_psi           = _rbm->log_wf(_spin, _sum_logcosh_theta);
    }

};


TCM_SWARM_END_NAMESPACE


#endif // TCM_SWARM_RBM_SPIN_HPP
