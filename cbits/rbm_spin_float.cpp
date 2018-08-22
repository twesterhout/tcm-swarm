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

#include "rbm_spin_float.hpp"

auto _tcm_Rbm::allocate_buffer(index_type const dim1)
    -> std::tuple<buffer_type, index_type>
{
    Expects(dim1 >= 0);
    auto [p, n1] = tcm::allocate_aligned_buffer<C, simd_abi>(
        gsl::narrow_cast<std::size_t>(dim1));
    return {std::move(p), gsl::narrow_cast<index_type>(n1)};
}

auto _tcm_Rbm::allocate_buffer(index_type const dim1, index_type const dim2)
    -> std::tuple<buffer_type, index_type, index_type>
{
    Expects(dim1 >= 0 && dim2 >= 0);
    auto [p, n1, n2] = tcm::allocate_aligned_buffer<C, simd_abi>(
        gsl::narrow_cast<std::size_t>(dim1),
        gsl::narrow_cast<std::size_t>(dim2));
    return {std::move(p), gsl::narrow_cast<index_type>(n1),
        gsl::narrow_cast<index_type>(n2)};
}

inline auto _tcm_Rbm::_allocate_buffers(
    index_type const size_visible, index_type const size_hidden)
{
    Expects(size_visible >= 0);
    Expects(size_hidden >= 0);

    static_assert(layout_weights() == tcm::mkl::Layout::ColMajor);
    std::tie(_weights, std::ignore, _ldim_weights) =
        allocate_buffer(size_visible, size_hidden);
    for (auto i = 0; i < size_visible; ++i) {
        Ensures((tcm::detail::is_aligned<
            Vc::memory_alignment_v<Vc::simd<R, simd_abi>>>(
            _data(_weights) + i * _ldim_weights)));
    }
    std::tie(_visible, std::ignore) = allocate_buffer(size_visible);
    std::tie(_hidden, std::ignore)  = allocate_buffer(size_hidden);

    Ensures(tcm::detail::is_aligned<alignment()>(_data(_visible)));
    Ensures(tcm::detail::is_aligned<alignment()>(_data(_hidden)));
    Ensures(tcm::detail::is_aligned<alignment()>(_data(_weights)));
}

_tcm_Rbm::_tcm_Rbm(index_type const size_visible, index_type const size_hidden)
    : _size_visible{size_visible}, _size_hidden{size_hidden}
{
    if (size_visible < 0)
        tcm::throw_with_trace(tcm::negative_size_error{"size_visible", size_visible});
    if (size_hidden < 0)
        tcm::throw_with_trace(tcm::negative_size_error{"size_hidden", size_hidden});
    _allocate_buffers(size_visible, size_hidden);
}

template <tcm::mkl::Layout Layout>
auto _tcm_Rbm::load_weights(gsl::span<C const> w) -> void
{
    for (index_type i = 0; i < size_hidden(); ++i) {
        for (index_type j = 0; j < size_visible(); ++j) {
            if constexpr (Layout == tcm::mkl::Layout::RowMajor) {
                weights(i, j) = w[size_visible() * i + j];
            }
            else {
                weights(i, j) = w[i + size_hidden() * j];
            }
        }
    }
}

template auto _tcm_Rbm::load_weights<tcm::mkl::Layout::RowMajor>(gsl::span<C const>)
    -> void;
template auto _tcm_Rbm::load_weights<tcm::mkl::Layout::ColMajor>(gsl::span<C const>)
    -> void;

auto _tcm_Rbm::load_visible(gsl::span<C const> const a) -> void
{
    using std::begin, std::end;
    Expects(a.size() == size_visible());
    std::copy_n(a.data(), size_visible(), _data(_visible));
}

auto _tcm_Rbm::load_hidden(gsl::span<C const> const b) -> void
{
    using std::begin, std::end;
    Expects(b.size() == size_hidden());
    std::copy_n(b.data(), size_hidden(), _data(_hidden));
}

