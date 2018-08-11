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
#include "memory.hpp"

#include "spin.hpp"

#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostics push
#pragma clang diagnostics ignore "-Wzero-as-null-pointer-constant"
#endif

#include <boost/core/demangle.hpp>

#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostics pop
#endif


#include "nqs_types.h"

TCM_SWARM_BEGIN_NAMESPACE

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
template <class T, class = void>
class RbmBase;

template <class T, class = void>
struct McmcBase;

template <class T, class Generator>
McmcBase(RbmBase<T> const&, Generator&)->McmcBase<T>;

template <class T, class Generator>
McmcBase(RbmBase<T> const&, Generator&, typename RbmBase<T>::index_type)
    ->McmcBase<T>;

template <class R>
class RbmBase<std::complex<R>, std::enable_if_t<std::is_floating_point_v<R>>> {

  private:
    using self_type = RbmBase; // <std::complex<R>, Allocator>;
    using C         = std::complex<R>;


  public:
    using value_type      = C;
    using index_type      = int;
    using size_type       = int;
    using difference_type = index_type;
    using buffer_type     = std::unique_ptr<C[], FreeDeleter>;
    using default_simd_vector = Vc::simd<R, Vc::simd_abi::native<R>>;

    static constexpr auto alignment() noexcept -> std::size_t
    {
        return detail::alignment<R>();
    }

  private:
    // TODO: Keep this in sync with tcm_PolyRbm !!!
    buffer_type  _weights; ///< Weights of the RBM.
    buffer_type  _visible; ///< Visible bias.
    buffer_type  _hidden;  ///< Hidden bias.
    index_type   _size_visible;
    index_type   _size_hidden;
    index_type   _ldim_weights;

    auto _allocate_buffers(
        index_type const size_visible, index_type const size_hidden)
    {
        Expects(size_visible >= 0);
        Expects(size_hidden >= 0);

        std::size_t ldim;
        if constexpr (layout_weights() == mkl::Layout::ColMajor) {
            std::tie(_weights, _ldim_weights) =
                allocate_aligned_buffer<C, alignment(),
                    default_simd_vector::size()>(size_visible, size_hidden);
        }
        else /*RowMajor*/ {
            std::tie(_weights, _ldim_weights) =
                allocate_aligned_buffer<C, alignment(),
                    default_simd_vector::size()>(size_hidden, size_visible);
        }
        _visible = allocate_aligned_buffer<C, alignment(),
            default_simd_vector::size()>(size_visible);
        _hidden  = allocate_aligned_buffer<C, alignment(),
            default_simd_vector::size()>(size_hidden);

        Ensures(TCM_SWARM_IS_ALIGNED(_weights.get(), detail::alignment<R>()));
        Ensures(TCM_SWARM_IS_ALIGNED(_visible.get(), detail::alignment<R>()));
        Ensures(TCM_SWARM_IS_ALIGNED(_hidden.get(), detail::alignment<R>()));
    }

    static auto*       _data(buffer_type& x) noexcept { return x.get(); }
    static auto const* _data(buffer_type const& x) noexcept { return x.get(); }

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
        : _size_visible{size_visible}
        , _size_hidden{size_hidden}
    {
        if (size_visible < 0)
            throw_with_trace(negative_size_error{"size_visible", size_visible});
        if (size_hidden < 0)
            throw_with_trace(negative_size_error{"size_hidden", size_hidden});
        _allocate_buffers(size_visible, size_hidden);
    }

    /// \defgroup RbmSpinCopyMove Copy and Move
    /// \{

    RbmBase(RbmBase const&)     = default;
    RbmBase(RbmBase&&) noexcept = default;
    RbmBase& operator=(RbmBase const&) = default;
    RbmBase& operator=(RbmBase&&) noexcept = default;

    /// \}

    explicit operator tcm_Rbm&() & noexcept
    {
        return *reinterpret_cast<tcm_Rbm*>(this);
    }

    explicit operator tcm_Rbm const&() const& noexcept
    {
        return *reinterpret_cast<tcm_Rbm const*>(this);
    }

    /// \brief Returns memory layout (column-major or row-major) of the weight matrix.
    static constexpr auto layout_weights() noexcept
    {
        return mkl::Layout::ColMajor;
    }

    /// \brief Returns number of visible units.
    constexpr auto size_visible() const noexcept -> index_type
    {
        return _size_visible;
    }

    /// \brief Returns number of hidden units.
    constexpr auto size_hidden() const noexcept -> index_type
    {
        return _size_hidden;
    }

    /// \brief Returns number of elements in the weight matrix.
    ///
    /// This is equivalent to ``size_visible() * size_hidden()``.
    constexpr auto size_weights() const noexcept -> index_type
    {
        return size_visible() * size_hidden();
    }

    /// \brief Returns the total number of parameters in the machine.
    ///
    /// This is equivalent to ``size_visible() + size_hidden() +
    /// size_weights()``.
    constexpr auto size() const noexcept -> index_type
    {
        return size_visible() + size_hidden() + size_weights();
    }

    constexpr auto ldim_weights() const noexcept -> index_type
    {
        return _ldim_weights;
    }

  private:
    // clang-format off
    template <class T>
    TCM_SWARM_FORCEINLINE
    static constexpr auto matrix_access(T* m, index_type const ldim,
        index_type const r, index_type const c)
        // clang-format on
        noexcept(!detail::gsl_can_throw()) -> T*
    {
        if constexpr (layout_weights() == mkl::Layout::ColMajor) {
            return m + r + ldim * c;
        }
        else {
            return m + ldim * r + c;
        }
    }

  public:
    /// \defgroup RbmSpinAccessors Getters and Setters
    /// \{

    /// \brief Returns \f$w_{rc}\f$.
    constexpr auto weights(index_type const r, index_type const c) const&
        noexcept(!detail::gsl_can_throw()) -> C
    {
        Expects(0 <= r && r < size_hidden());
        Expects(0 <= c && c < size_visible());
        return *matrix_access(_data(_weights), ldim_weights(), r, c);
    }

    /// \overload
    constexpr auto weights(index_type const r, index_type const c)
        & noexcept(!detail::gsl_can_throw()) -> C&
    {
        Expects(0 <= r && r < size_hidden());
        Expects(0 <= c && c < size_visible());
        return *matrix_access(_data(_weights), ldim_weights(), r, c);
    }

    /// \overload
    ///
    /// Returns `[weights(r, c), weights(r+1, c), ..., weights(r+N-1, c)]` where
    /// `N == Vc::simd<R, Abi>::size()`.
    ///
    /// .. warning::
    ///
    ///    ``r`` must be a multiple of ``Vc::simd<R, Abi>::size()``.
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto weights_v(index_type const r, index_type const c,
        Vc::flags::vector_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        static_assert(vector_size % 2 == 0);
        static_assert(layout_weights() == mkl::Layout::ColMajor);
        Expects(0 <= r && r < size_hidden());
        Expects(0 <= c && c < size_visible());
        Expects((r & (vector_size - 1)) == 0);
        auto const* p = matrix_access(_data(_weights), ldim_weights(), r, c);
        Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
        return copy_from<Abi>(p, flag);
    }

    /// \brief Returns visible bias.
    constexpr auto visible() const & noexcept -> gsl::span<C const>
    {
        return {_data(_visible), _size_visible};
    }

    /// \overload
    constexpr auto visible() & noexcept -> gsl::span<C>
    {
        return {_data(_visible), _size_visible};
    }

    /// \brief Returns \f$a_i\f$
    constexpr auto visible(index_type const i) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        Expects(0 <= i && i < size_visible());
        return _visible[i];
    }

    /// \overload
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto visible_v(
        index_type const i, Vc::flags::vector_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        static_assert(vector_size % 2 == 0);
        Expects(0 <= i && i < size_visible());
        Expects((i & (vector_size - 1)) == 0);
        auto const* p = _data(_visible) + i;
        Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
        return copy_from<Abi>(p, flag);
    }

    /// \brief Returns hidden bias.
    constexpr auto hidden() const& noexcept -> gsl::span<C const>
    {
        return {_data(_hidden), _size_hidden};
    }

    /// \overload
    constexpr auto hidden() & noexcept -> gsl::span<C>
    {
        return {_data(_hidden), _size_hidden};
    }

    /// \brief Returns \f$b_i\f$
    constexpr auto hidden(index_type const i) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        Expects(0 <= i && i < size_hidden());
        return _hidden[i];
    }

    /// \overload
    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto hidden_v(
        index_type const i, Vc::flags::vector_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        static_assert(vector_size % 2 == 0);
        Expects(0 <= i && i < size_hidden());
        Expects((i & (vector_size - 1)) == 0);
        auto const* p = _data(_hidden) + i;
        Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
        return copy_from<Abi>(p, flag);
    }

    /// \}

    // TODO: Purely for testing. Don't you dare use this function in real code!
    template <mkl::Layout Layout>
    auto load_weights(gsl::span<C const> w)
    {
        for (index_type i = 0; i < size_hidden(); ++i) {
            for (index_type j = 0; j < size_visible(); ++j) {
                if constexpr (Layout == mkl::Layout::RowMajor) {
                    weights(i, j) = w[size_visible() * i + j];
                }
                else {
                    weights(i, j) = w[i + size_hidden() * j];
                }
            }
        }
    }

    auto load_visible(gsl::span<C const> const a) noexcept(
        !detail::gsl_can_throw())
    {
        using std::begin, std::end;
        Expects(a.size() == size_visible());
        std::copy_n(a.data(), size_visible(), _data(_visible));
    }

    auto load_hidden(gsl::span<C const> const b) noexcept(
        !detail::gsl_can_throw())
    {
        using std::begin, std::end;
        Expects(b.size() == size_hidden());
        std::copy_n(b.data(), size_hidden(), _data(_hidden));
    }

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
    auto theta(gsl::span<C const> const spin, gsl::span<C> const out) const
        noexcept(!detail::gsl_can_throw()) -> void
    {
        using std::begin, std::end;
        Expects(spin.size() == size_visible());
        Expects(out.size() == size_hidden());
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));
        // theta := b
        auto const b = hidden();
        std::copy(begin(b), end(b), begin(out));
        // theta := 1.0 * w * spin + 1.0 * theta
        constexpr mkl::difference_type one = 1;
        mkl::gemv(layout_weights(), mkl::Transpose::None, size_hidden(),
            size_visible(), C{1}, _data(_weights), ldim_weights(), spin.data(),
            one, C{1}, out.data(), one);
    }

    /// \overload
    auto theta(gsl::span<C const> const spin) const -> buffer_type
    {
        Expects(spin.size() == size_visible());
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));
        auto       out_buffer = allocate_aligned_buffer<C, alignment(),
            default_simd_vector::size()>(size_hidden());
        auto const out = gsl::make_span<C>(_data(out_buffer), size_hidden());
        theta(spin, out);
        return out_buffer;
    }

    auto log_wf(gsl::span<C const> const spin, C const sum_logcosh_theta) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        constexpr R log_of_2{0.6931471805599453094172321214581765680755};
        Expects(spin.size() == size_visible());
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));
        return gsl::narrow_cast<R>(size_hidden()) * log_of_2
               + mkl::dotu(visible(), spin) + sum_logcosh_theta;
    }

    /// \brief Calculates \f$\log\psi(\sigma; \mathcal{W})\f$.
    ///
    /// This function is implemented as
    /// \f[
    ///     \log\psi(\sigma; \mathcal{W}) = \operatorname{Re}[a]\cdot\sigma
    ///         + \operatorname{Im}[a]\cdot\sigma + \sum_i\cosh(\theta) \;.
    /// \f]
    auto log_wf(gsl::span<C const> const spin) const -> C
    {
        Expects(spin.size() == size_visible());
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));

        auto const theta_buffer      = theta(spin);
        auto const sum_logcosh_theta = sum_log_cosh(
            gsl::make_span<C const>(_data(theta_buffer), size_hidden()));
        return log_wf(spin, sum_logcosh_theta);
    }

    auto der_log_wf(gsl::span<C const> const spin,
        gsl::span<C const> const theta, gsl::span<C> const out) const
        noexcept(!detail::gsl_can_throw())
    {
        using std::begin, std::end;
        Expects(spin.size() == size_visible());
        Expects(theta.size() == size_hidden());
        Expects(out.size() == size());
        Expects(TCM_SWARM_IS_ALIGNED(spin.data(), detail::alignment<R>()));
        Expects(TCM_SWARM_IS_ALIGNED(theta.data(), detail::alignment<R>()));
        Expects(TCM_SWARM_IS_ALIGNED(out.data(), detail::alignment<R>()));
        Expects(TCM_SWARM_IS_VALID_SPIN(spin));

        std::copy(begin(spin), end(spin), begin(out));
        mkl::tanh(theta, out.subspan(spin.size(), theta.size()));
        gsl::span<C const> const spin_part = out.subspan(0, spin.size());
        gsl::span<C const> const tanh_part =
            out.subspan(spin.size(), theta.size());
        mkl::geru(C{1.0}, tanh_part, spin_part,
            out.subspan(spin.size() + theta.size(), spin.size() * theta.size()),
            layout_weights());
        mkl::scale(R{0.5}, out);
    }
};

/// \brief This class caches a few extra things which allow for faster
/// construction of Markov chains, but are not really part of the Rbm.
template <class R>
struct McmcBase<std::complex<R>,
    std::enable_if_t<std::is_floating_point_v<R>>> {

  private:
    using C         = std::complex<R>;
    using Rbm       = tcm::RbmBase<C>;
    using self_type = McmcBase<C>;

    using default_simd_vector = Vc::simd<R, Vc::simd_abi::native<R>>;

  public:
    using value_type      = typename Rbm::value_type;
    using index_type      = typename Rbm::index_type;
    using size_type       = typename Rbm::size_type;
    using difference_type = typename Rbm::difference_type;
    using buffer_type     = typename Rbm::buffer_type;

  private:
    gsl::not_null<Rbm const*> _rbm;
    buffer_type _spin;  ///< Current spin configuration \f$\sigma\f$.
    buffer_type _theta; ///< Cached \f$\theta\f$ (i.e. \f$b + w \sigma\f$).
    C           _sum_logcosh_theta; ///< Cached \f$\sum_i\log\cosh(\theta_i)\f$.
    C           _log_psi; ///< Cached \f$\log\Psi_\mathcal{W}(\sigma)\f$.

    static auto*       _data(buffer_type& x) noexcept { return x.get(); }
    static auto const* _data(buffer_type const& x) noexcept { return x.get(); }

    auto const* _as_c() const noexcept { return this; }

    static constexpr auto alignment() noexcept { return Rbm::alignment(); }

    static auto buffer_from_span(gsl::span<C const> x) -> buffer_type
    {
        using std::begin, std::end;
        auto x_buffer = allocate_aligned_buffer(x.size());
        std::copy(begin(x), end(x), _data(x_buffer));
        return x_buffer;
    }

    auto initialise()
    {
        _rbm->theta(_as_c()->spin(), theta());
        _sum_logcosh_theta = sum_log_cosh(_as_c()->theta());
        _log_psi           = _rbm->log_wf(_as_c()->spin(), _sum_logcosh_theta);
    }

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
    McmcBase(Rbm const& rbm, buffer_type&& initial_spin)
        : _rbm{std::addressof(rbm)}
        , _spin{std::move(initial_spin)}
        , _theta{allocate_aligned_buffer<C, alignment(),
              default_simd_vector::size()>(rbm.size_hidden())}
    {
        Expects(TCM_SWARM_IS_ALIGNED(_data(_spin), detail::alignment<R>()));
        Expects(TCM_SWARM_IS_VALID_SPIN(spin()));
        initialise();
    }

    /// \overload
    McmcBase(Rbm const& rbm, gsl::span<C const> const spin)
        : McmcBase{rbm, buffer_from_span(spin)}
    {
    }

    template <class Generator>
    McmcBase(Rbm const& rbm, Generator& gen)
        : _rbm{std::addressof(rbm)}
        , _spin{allocate_aligned_buffer<C, alignment(),
              default_simd_vector::size()>(rbm.size_visible())}
        , _theta{allocate_aligned_buffer<C, alignment(),
              default_simd_vector::size()>(rbm.size_hidden())}
    {
        using std::begin, std::end;
        auto                               spin = this->spin();
        std::uniform_int_distribution<int> dist{0, 1};
        std::generate(begin(spin), end(spin), [&gen, &dist]() -> C {
            return {R{2} * gsl::narrow<R>(dist(gen)) - R{1}, 0};
        });
        Ensures(TCM_SWARM_IS_ALIGNED(_data(_spin), alignment()));
        Ensures(TCM_SWARM_IS_VALID_SPIN(_as_c()->spin()));
        initialise();
    }

    template <class Generator>
    McmcBase(Rbm const& rbm, Generator& gen, index_type const magnetisation)
        : _rbm{std::addressof(rbm)}
        , _spin{allocate_aligned_buffer<C, alignment(),
              default_simd_vector::size()>(rbm.size_visible())}
        , _theta{allocate_aligned_buffer<C, alignment(),
              default_simd_vector::size()>(rbm.size_hidden())}
    {
        using std::begin, std::end;
        if (std::abs(magnetisation) > rbm.size_visible()) {
            std::ostringstream msg;
            msg << rbm.size_visible()
                << " spins can't have a total magnetisation of "
                << magnetisation << ".";
            throw_with_trace(std::domain_error{msg.str()});
        }
        if ((rbm.size_visible() + magnetisation) % 2 != 0) {
            std::ostringstream msg;
            msg << rbm.size_visible()
                << " spins can't have a total magnetisation of "
                << magnetisation
                << ". `size_visible() + magnetisation` must be even.";
            throw_with_trace(std::domain_error{msg.str()});
        }
        auto const n            = size_visible();
        auto const number_ups   = (n + magnetisation) / 2;
        auto const number_downs = (n - magnetisation) / 2;
        auto       spin         = this->spin();
        std::fill_n(begin(spin), number_ups, C{1});
        std::fill_n(begin(spin) + number_ups, number_downs, C{-1});
        std::shuffle(begin(spin), end(spin), gen);
        Ensures(TCM_SWARM_IS_ALIGNED(_data(_spin), alignment()));
        Ensures(TCM_SWARM_IS_VALID_SPIN(spin));
        Ensures(TCM_SWARM_CHECK_MAGNETISATION(_as_c()->spin(), magnetisation));
        initialise();
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

    constexpr auto size() const noexcept { return _rbm->size(); }

    constexpr auto theta() const & noexcept -> gsl::span<C const>
    {
        return {_data(_theta), size_hidden()};
    }

    constexpr auto theta() & noexcept -> gsl::span<C>
    {
        return {_data(_theta), size_hidden()};
    }

    template <class Abi = Vc::simd_abi::native<R>>
    constexpr auto theta_v(
        index_type const i, Vc::flags::vector_aligned_tag flag) const
        noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        static_assert(vector_size % 2 == 0);
        Expects(0 <= i && i < size_hidden());
        Expects((i & (vector_size - 1)) == 0);
        auto const* p = _data(_theta) + i;
        Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
        return copy_from<Abi>(p, flag);
    }

    constexpr auto spin() const& noexcept -> gsl::span<C const>
    {
        return {_data(_spin), size_visible()};
    }

    constexpr auto spin() & noexcept -> gsl::span<C>
    {
        return {_data(_spin), size_visible()};
    }

    constexpr auto spin(index_type const i) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        Expects(0 <= i && i < size_visible());
        return _spin[i];
    }

    decltype(auto) spin(gsl::span<C const> const new_spin) noexcept(
        !detail::gsl_can_throw())
    {
        using std::begin, std::end;
        Expects(new_spin.size() == size_visible());
        Expects(TCM_SWARM_IS_VALID_SPIN(new_spin));
        std::copy(begin(new_spin), end(new_spin), _data(_spin));
        initialise();
        return *this;
    }

    decltype(auto) spin(buffer_type&& new_spin) noexcept(
        !detail::gsl_can_throw())
    {
        Expects(TCM_SWARM_IS_ALIGNED(_data(_spin), detail::alignment<R>()));
        Expects(TCM_SWARM_IS_VALID_SPIN((
            gsl::span<C const>{_data(new_spin), size_visible()})));
        _spin = std::move(new_spin);
        initialise();
        return *this;
    }

  private:
    template <std::ptrdiff_t Extent>
    auto flips_are_within_bounds(
        gsl::span<index_type const, Extent> const flips) const noexcept -> bool
    {
#if !defined(TCM_SWARM_NOCHECK_FLIPS_BOUNDS)
        using std::begin, std::end;
        return std::all_of(begin(flips), end(flips),
            [n = size_visible()](auto const f) { return 0 <= f && f < n; });
#else
        return true;
#endif
    }

    template <std::ptrdiff_t Extent>
    auto flips_are_unique(gsl::span<index_type const, Extent> const flips) const
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
    // clang-format off
    template <std::ptrdiff_t Extent>
    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto new_theta(index_type const i,
        gsl::span<index_type const, Extent> const flips) const
        // clang-format on
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

    // clang-format off
    template <std::ptrdiff_t Extent, class Abi = Vc::simd_abi::native<R>>
    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto new_theta_v(index_type const i,
        gsl::span<index_type const, Extent> const flips,
        Vc::flags::vector_aligned_tag flag) const
    // clang-format on
        noexcept(!detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
    {
        using V = Vc::simd<R, Abi>;
        Expects(flips_are_within_bounds(flips));
        Expects(flips_are_unique(flips));

        std::complex<V> delta{V{0}, V{0}};
        for (auto const flip : flips) {
            delta += _rbm->weights_v(i, flip, Vc::flags::vector_aligned)
                     * V{_spin[flip].real()};
        }
        return theta_v(i, flag) - V{2} * delta;
    }

    // clang-format off
    template <std::ptrdiff_t Extent>
    TCM_SWARM_ARTIFIFICAL
    TCM_SWARM_FORCEINLINE
    TCM_SWARM_PURE
    auto sum_logcosh_new_theta(gsl::span<index_type const> const flips) const
        // clang-format on
        noexcept(!detail::gsl_can_throw()) -> C
    {
        C sum{0};
        for (size_type i = 0, n = size_hidden(); i < n; ++i) {
            sum += _log_cosh(new_theta(i, flips));
        }
        return sum;
    }

    template <std::ptrdiff_t Extent, class Abi = Vc::simd_abi::native<R>>
    auto sum_logcosh_new_theta_v(
        gsl::span<index_type const, Extent> const flips) const
        noexcept(!detail::gsl_can_throw()) -> C
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());

        std::complex<V> sum{V{0}, V{0}};
        for (index_type i = 0; i < size_hidden(); i += vector_size) {
            sum += _log_cosh(new_theta_v(i, flips, Vc::flags::vector_aligned));
        }

        return std::complex{Vc::reduce(sum.real()), Vc::reduce(sum.imag())};
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
    template <std::ptrdiff_t Extent>
    auto log_quot_wf(gsl::span<index_type const, Extent> const flips) const
        noexcept(!detail::gsl_can_throw()) -> std::tuple<C, Cache>
    {
        Expects(flips_are_within_bounds(flips));
        Expects(flips_are_unique(flips));

        auto const sum_logcosh_new = sum_logcosh_new_theta_v(flips);
        C          delta{0};
        for (auto flip : flips) {
            delta += _rbm->visible(flip) * _spin[flip];
        }
        auto const log_quot_wf =
            sum_logcosh_new - _sum_logcosh_theta - C{2} * delta;
        return {log_quot_wf, {sum_logcosh_new}};
    }

    constexpr auto log_wf() const noexcept -> C { return _log_psi; }

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

    template <std::ptrdiff_t Extent, class Abi = Vc::simd_abi::native<R>>
    auto
    update_theta_v(gsl::span<index_type const, Extent> const flips) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        using V                    = Vc::simd<R, Abi>;
        constexpr auto vector_size = static_cast<index_type>(V::size());
        auto*          data        = _data(_theta);
        for (index_type i = 0; i < size_hidden();
             i += vector_size, data += vector_size) {
            copy_to(new_theta_v(i, flips, Vc::flags::vector_aligned), data,
                Vc::flags::vector_aligned);
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
        Ensures(TCM_SWARM_IS_VALID_SPIN(_as_c()->spin()));
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
        _log_psi = _rbm->log_wf(spin(), _sum_logcosh_theta);
    }

    template <std::ptrdiff_t Extent>
    auto update(gsl::span<index_type const, Extent> const flips) noexcept(
        !detail::gsl_can_throw()) -> void
    {
        update_theta_v(flips);
        update_spin(flips);
        _sum_logcosh_theta = sum_log_cosh(_as_c()->theta());
        _log_psi           = _rbm->log_wf(spin(), _sum_logcosh_theta);
    }

    template <std::ptrdiff_t Extent>
    auto der_log_wf(gsl::multi_span<C, Extent> const out) const
        noexcept(!detail::gsl_can_throw())
    {
        der_log_wf(gsl::span<C, Extent>{out.data(), out.size()});
    }

    template <std::ptrdiff_t Extent>
    auto der_log_wf(gsl::span<C, Extent> const out) const
        noexcept(!detail::gsl_can_throw())
    {
        _rbm->der_log_wf(spin(), theta(), out);
    }
};

TCM_SWARM_END_NAMESPACE


#endif // TCM_SWARM_RBM_SPIN_HPP
