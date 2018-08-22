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

#ifndef TCM_SWARM_RBM_SPIN_FLOAT_HPP
#define TCM_SWARM_RBM_SPIN_FLOAT_HPP

#include "detail/config.hpp"
#include "detail/mkl.hpp"
#include "mcmc_state.hpp"
#include "memory.hpp"
#include "nqs_types.h"

#include <complex>
#include <memory>

#include <Vc/Vc>
#include <gsl/span>

struct TCM_SWARM_SYMBOL_VISIBLE _tcm_Rbm {
  public:
    using R                   = float;
    using C                   = std::complex<R>;
    using value_type          = C;
    using index_type          = std::ptrdiff_t;
    using size_type           = index_type;
    using difference_type     = index_type;
    using buffer_type         = std::unique_ptr<C[], tcm::FreeDeleter>;
    using simd_abi            = Vc::simd_abi::native<R>;

    // This makes a conversion from non-negative index_type to std::size_t safe.
    static_assert(sizeof(index_type) <= sizeof(std::size_t));

    static constexpr auto alignment() noexcept -> std::size_t
    {
        return tcm::detail::alignment<R>();
    }

  private:
    buffer_type _weights; ///< Weights of the RBM.
    buffer_type _visible; ///< Visible bias.
    buffer_type _hidden;  ///< Hidden bias.
    index_type  _size_visible;
    index_type  _size_hidden;
    index_type  _ldim_weights;

    auto _allocate_buffers(index_type size_visible, index_type size_hidden);

    // TODO(twesterhout): It's really annoying that std::unique_ptr::get is not
    // constexpr!
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
    _tcm_Rbm(index_type size_visible, index_type size_hidden);

    /// \defgroup RbmSpinCopyMove Copy and Move
    /// \{

    _tcm_Rbm(_tcm_Rbm const&)     = default;
    _tcm_Rbm(_tcm_Rbm&&) noexcept = default;
    _tcm_Rbm& operator=(_tcm_Rbm const&) = default;
    _tcm_Rbm& operator=(_tcm_Rbm&&) noexcept = default;

    /// \}

    /// \brief Returns memory layout (column-major or row-major) of the weight matrix.
    static constexpr auto layout_weights() noexcept
    {
        return tcm::mkl::Layout::ColMajor;
    }

    constexpr auto ldim_weights() const noexcept -> index_type
    {
        return _ldim_weights;
    }

    // clang-format off
    constexpr auto size_visible() const noexcept { return _size_visible; }
    constexpr auto size_hidden() const noexcept { return _size_hidden; }
    constexpr auto size_weights() const noexcept { return size_visible() * size_hidden(); }
    // clang-format on

    constexpr auto size() const noexcept
    {
        return size_visible() + size_hidden() + size_weights();
    }

    static auto allocate_buffer(index_type)
        -> std::tuple<buffer_type, index_type>;
    static auto allocate_buffer(index_type, index_type)
        -> std::tuple<buffer_type, index_type, index_type>;

  private:
    // clang-format off
    template <class T>
    TCM_SWARM_FORCEINLINE
    static constexpr auto matrix_access(T* m, index_type const ldim,
        index_type const r, index_type const c) noexcept(!tcm::detail::gsl_can_throw())
            -> T*
    // clang-format on
    {
        if constexpr (layout_weights() == tcm::mkl::Layout::ColMajor) {
            return m + r + ldim * c;
        }
        else {
            return m + ldim * r + c;
        }
    }

  public:
    /// \defgroup RbmSpinAccessors Getters and Setters
    /// \{

    auto weights() noexcept -> tcm_Matrix;
    auto visible() noexcept -> tcm_Vector;
    auto hidden() noexcept -> tcm_Vector;

  // private:
    auto visible_span() const noexcept -> gsl::span<C const>;
    auto visible_span() noexcept -> gsl::span<C>;
    auto hidden_span() const noexcept -> gsl::span<C const>;
    auto hidden_span() noexcept -> gsl::span<C>;

  public:
    auto weights(index_type r, index_type c) const
        noexcept(!tcm::detail::gsl_can_throw()) -> C;

    auto visible(index_type i) const
        noexcept(!tcm::detail::gsl_can_throw()) -> C;

    auto hidden(index_type i) const noexcept(!tcm::detail::gsl_can_throw())
        -> C;


    auto weights(index_type r, index_type c) noexcept(
        !tcm::detail::gsl_can_throw()) -> C&;

    auto visible(index_type i) noexcept(!tcm::detail::gsl_can_throw())
        -> C&;

    auto hidden(index_type i) noexcept(!tcm::detail::gsl_can_throw())
        -> C&;

    /// \overload
    ///
    /// Returns `[weights(r, c), weights(r+1, c), ..., weights(r+N-1, c)]` where
    /// `N == Vc::simd<R, Abi>::size()`.
    ///
    /// .. warning::
    ///
    ///    ``r`` must be a multiple of ``Vc::simd<R, Abi>::size()``.
    template <class Abi = Vc::simd_abi::native<R>>
    auto weights_v(
        index_type r, index_type c, Vc::flags::vector_aligned_tag flag) const
        noexcept(!tcm::detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>;

    template <class Abi = Vc::simd_abi::native<R>>
    auto visible_v(
        index_type i, Vc::flags::vector_aligned_tag flag) const
        noexcept(!tcm::detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>;

    template <class Abi = Vc::simd_abi::native<R>>
    auto hidden_v(
        index_type i, Vc::flags::vector_aligned_tag flag) const
        noexcept(!tcm::detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>;

    auto theta(gsl::span<C const> spin, gsl::span<C> out) const
        noexcept(!tcm::detail::gsl_can_throw()) -> void;

    auto theta(gsl::span<C const> spin) const -> buffer_type;

    auto log_wf(gsl::span<C const> spin, C sum_log_cosh_theta) const
        noexcept(!tcm::detail::gsl_can_throw()) -> C;

    auto log_wf(gsl::span<C const> spin) const -> C;

    auto der_log_wf(gsl::span<C const> spin, gsl::span<C const> theta,
        gsl::span<C> out) const noexcept(!tcm::detail::gsl_can_throw()) -> void;

    // TODO: Purely for testing. Don't you dare use these functions in real code!
    template <tcm::mkl::Layout Layout>
    auto load_weights(gsl::span<C const> w) -> void;
    auto load_visible(gsl::span<C const> a) -> void;
    auto load_hidden(gsl::span<C const> b) -> void;

    template <class Generator>
    auto make_state(std::optional<index_type> magnetisation,
        Generator& generator) const -> std::unique_ptr<tcm::McmcState>;
};

TCM_SWARM_BEGIN_NAMESPACE

using Rbm = _tcm_Rbm;

TCM_SWARM_END_NAMESPACE

// And the implementation
#include "rbm_spin_float.ipp"

#endif // TCM_SWARM_RBM_SPIN_FLOAT_HPP
