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

#ifndef TCM_SWARM_RBM_SPIN_FLOAT_IPP
#define TCM_SWARM_RBM_SPIN_FLOAT_IPP

// This is not, strictly speaking required, but helps the IDEs.
#include "rbm_spin_float.hpp"

#include "detail/axpby.hpp"
#include "detail/dotu.hpp"
#include "detail/errors.hpp"
#include "detail/gemv.hpp"
#include "detail/geru.hpp"
// #include "detail/lncosh.hpp"
// #include "detail/mkl_allocator.hpp"
// #include "detail/random.hpp"
#include "detail/scale.hpp"
#include "detail/simd.hpp"
#include "detail/tanh.hpp"
#include "memory.hpp"
#include "spin_utils.hpp"

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::weights() noexcept -> tcm_Matrix
{
    return {_data(_weights), gsl::narrow_cast<int>(size_hidden()),
        gsl::narrow_cast<int>(size_visible()),
        gsl::narrow_cast<int>(ldim_weights())};
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::visible() noexcept -> tcm_Vector
{
    return {_data(_visible), gsl::narrow_cast<int>(size_visible()), 1};
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::hidden() noexcept -> tcm_Vector
{
    return {_data(_hidden), gsl::narrow_cast<int>(size_hidden()), 1};
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::weights(index_type const r, index_type const c) const
    noexcept(!tcm::detail::gsl_can_throw()) -> C
{
    Expects(0 <= r && r < size_hidden());
    Expects(0 <= c && c < size_visible());
    return *matrix_access(_data(_weights), ldim_weights(), r, c);
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::visible(index_type const i) const
    noexcept(!tcm::detail::gsl_can_throw()) -> C
{
    Expects(0 <= i && i < size_visible());
    return _visible[gsl::narrow_cast<std::size_t>(i)];
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::hidden(index_type const i) const
    noexcept(!tcm::detail::gsl_can_throw()) -> C
{
    Expects(0 <= i && i < size_hidden());
    return _hidden[gsl::narrow_cast<std::size_t>(i)];
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::weights(index_type const r,
    index_type const c) noexcept(!tcm::detail::gsl_can_throw()) -> C&
{
    Expects(0 <= r && r < size_hidden());
    Expects(0 <= c && c < size_visible());
    return *matrix_access(_data(_weights), ldim_weights(), r, c);
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::visible(index_type const i) noexcept(
    !tcm::detail::gsl_can_throw()) -> C&
{
    Expects(0 <= i && i < size_visible());
    return _visible[gsl::narrow_cast<std::size_t>(i)];
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::hidden(index_type const i) noexcept(
    !tcm::detail::gsl_can_throw()) -> C&
{
    Expects(0 <= i && i < size_hidden());
    return _hidden[gsl::narrow_cast<std::size_t>(i)];
}

template <class Abi>
TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::weights_v(index_type const r, index_type const c,
    Vc::flags::vector_aligned_tag const flag) const
    noexcept(!tcm::detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
{
    using V                    = Vc::simd<R, Abi>;
    constexpr auto vector_size = static_cast<index_type>(V::size());
    static_assert(vector_size % 2 == 0);
    static_assert(layout_weights() == tcm::mkl::Layout::ColMajor);
    Expects(0 <= r && r < size_hidden());
    Expects(0 <= c && c < size_visible());
    Expects((r & (vector_size - 1)) == 0);
    // std::cerr << "_tcm_Rbm::weights_v(" << r << ", " << c << ").\n" << std::flush;
    auto const* p = matrix_access(_data(_weights), ldim_weights(), r, c);
    Expects(tcm::detail::is_aligned<Vc::memory_alignment_v<V>>(p));
    return tcm::copy_from<Abi>(p, flag);
}

template <class Abi>
TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::visible_v(
    index_type const i, Vc::flags::vector_aligned_tag const flag) const
    noexcept(!tcm::detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
{
    using V                    = Vc::simd<R, Abi>;
    constexpr auto vector_size = static_cast<index_type>(V::size());
    static_assert(vector_size % 2 == 0);
    Expects(0 <= i && i < size_visible());
    Expects((i & (vector_size - 1)) == 0);
    auto const* p = _data(_visible) + i;
    Expects(tcm::detail::is_aligned<Vc::memory_alignment_v<V>>(p));
    return tcm::copy_from<Abi>(p, flag);
}

template <class Abi>
TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::hidden_v(
    index_type const i, Vc::flags::vector_aligned_tag const flag) const
    noexcept(!tcm::detail::gsl_can_throw()) -> std::complex<Vc::simd<R, Abi>>
{
    using V                    = Vc::simd<R, Abi>;
    constexpr auto vector_size = static_cast<index_type>(V::size());
    static_assert(vector_size % 2 == 0);
    Expects(0 <= i && i < size_hidden());
    Expects((i & (vector_size - 1)) == 0);
    auto const* p = _data(_hidden) + i;
    Expects(TCM_SWARM_IS_ALIGNED(p, Vc::memory_alignment_v<V>));
    return tcm::copy_from<Abi>(p, flag);
}

inline auto _tcm_Rbm::theta(gsl::span<C const> spin, gsl::span<C> out) const
    noexcept(!tcm::detail::gsl_can_throw()) -> void
{
    using std::begin, std::end;
    Expects(spin.size() == size_visible());
    Expects(out.size() == size_hidden());
    Expects(tcm::is_valid_spin(spin));

    // theta := b
    // TODO: Rewrite it using Ranges TS syntax?
    auto const b = hidden_span();
    std::copy(begin(b), end(b), begin(out));

    // theta := 1.0 * w * spin + 1.0 * theta
    tcm::mkl::gemv(layout_weights(), tcm::mkl::Transpose::None, size_hidden(),
        size_visible(), C{1}, _data(_weights), ldim_weights(), spin.data(), 1,
        C{1}, out.data(), 1);
}

inline auto _tcm_Rbm::theta(gsl::span<C const> const spin) const -> buffer_type
{
    Expects(spin.size() == size_visible());
    Expects(tcm::is_valid_spin(spin));
    auto [out_buffer, _ignored_] = allocate_buffer(size_hidden());
    theta(spin, gsl::span{_data(out_buffer), size_hidden()});
    return std::move(out_buffer);
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::log_wf(gsl::span<C const> const spin, C const sum_log_cosh_theta) const
    noexcept(!tcm::detail::gsl_can_throw()) -> C
{
    constexpr R log_of_2{0.6931471805599453094172321214581765680755};
    Expects(spin.size() == size_visible());
    Expects(tcm::is_valid_spin(spin));
    return gsl::narrow_cast<R>(size_hidden()) * log_of_2
           + tcm::mkl::dotu(visible_span(), spin) + sum_log_cosh_theta;
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::log_wf(gsl::span<C const> const spin) const -> C
{
    Expects(spin.size() == size_visible());
    Expects(tcm::is_valid_spin(spin));
    auto const theta_buffer       = theta(spin);
    auto const sum_log_cosh_theta = tcm::sum_log_cosh(
        gsl::make_span<C const>(_data(theta_buffer), size_hidden()));
    return log_wf(spin, sum_log_cosh_theta);
}

inline auto _tcm_Rbm::der_log_wf(gsl::span<C const> const spin,
    gsl::span<C const> const theta, gsl::span<C> const out) const
    noexcept(!tcm::detail::gsl_can_throw()) -> void
{
    using std::begin, std::end;
    Expects(spin.size() == size_visible());
    Expects(theta.size() == size_hidden());
    Expects(out.size() == size());
    Expects(tcm::detail::is_aligned<alignment()>(spin.data()));
    Expects(tcm::detail::is_aligned<alignment()>(theta.data()));
    Expects(tcm::detail::is_aligned<alignment()>(out.data()));
    Expects(tcm::is_valid_spin(spin));

    std::copy(begin(spin), end(spin), begin(out));
    tcm::mkl::tanh(theta, out.subspan(spin.size(), theta.size()));

    auto prod_part =
        out.subspan(spin.size() + theta.size(), spin.size() * theta.size());
    gsl::span<C const> const spin_part = out.subspan(0, spin.size());
    gsl::span<C const> const tanh_part = out.subspan(spin.size(), theta.size());
    // WARNING: This call to memset is important, because geru does
    // ``A <- αXY' + A`` rather than ``A <- αXY'``
    std::fill_n(prod_part.data(), prod_part.size(), 0);
    tcm::mkl::geru(C{1}, tanh_part, spin_part, prod_part, layout_weights());
    // TODO: Measure, does this call matter?
    tcm::mkl::scale(R{0.5}, out);
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::visible_span() const noexcept -> gsl::span<C const>
{
    return {_data(_visible), size_visible()};
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::visible_span() noexcept -> gsl::span<C>
{
    return {_data(_visible), size_visible()};
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::hidden_span() const noexcept -> gsl::span<C const>
{
    return {_data(_hidden), size_hidden()};
}

TCM_SWARM_FORCEINLINE
auto _tcm_Rbm::hidden_span() noexcept -> gsl::span<C>
{
    return {_data(_hidden), size_hidden()};
}

#endif // TCM_SWARM_RBM_SPIN_FLOAT_IPP
