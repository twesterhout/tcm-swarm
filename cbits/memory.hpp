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

#pragma once
#include "detail/config.hpp"
#include "detail/errors.hpp"
#include "logging.hpp"
#include <boost/core/demangle.hpp>
#include <Vc/Vc>
#include <memory>

TCM_SWARM_BEGIN_NAMESPACE

struct FreeDeleter {
    template <class T>
    auto operator()(T* p) const noexcept -> void
    {
        std::free(p);
    }
};

namespace detail {
template <class Int, class = std::enable_if_t<std::is_integral_v<Int>>>
inline constexpr auto is_power_of_2(Int const n) noexcept -> bool
{
    return n > 0 && (n & (n - 1)) == 0;
}

/// \brief Determines which alignment to use for internal vectors (biases and
/// weights).
template <class T, class Abi = Vc::simd_abi::native<T>>
constexpr auto alignment() noexcept -> std::size_t
{
    // Intel MKL suggests to use buffers aligned to at least 64 bytes for
    // optimal performance.
    return std::max<std::size_t>(Vc::memory_alignment_v<Vc::simd<T, Abi>>, 64u);
}

/// \brief Rounds `n` up to a multiple of `Alignment`.
template <std::size_t Alignment>
constexpr auto round_up_to(std::size_t const n) noexcept -> std::size_t
{
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
        "Alignment must be a power of 2.");
    return (n + Alignment - 1) & ~(Alignment - 1);
}

template <class T>
[[noreturn]] auto integer_overflow_in_allocation(
    std::size_t dim, std::size_t count, std::size_t max_size) -> void
{
    std::string msg =
        format(fmt("Integer overflow encountered when trying to allocate "
                   "space for a vector of {} elements. The length was "
                   "rounded up to {} which exceeds {}, the maximal allowed "
                   "length for a buffer of {}."),
            dim, count, max_size, boost::core::demangle(typeid(T).name()));
    global_logger()->critical(msg);
    throw_with_trace(std::length_error{"Integer overflow."});
}

// clang-format off
template <class T, std::size_t Alignment, std::size_t VectorSize>
TCM_SWARM_FORCEINLINE
auto allocate_aligned_buffer(std::size_t const dim)
    -> std::tuple<std::unique_ptr<T[], FreeDeleter>, std::size_t>
// clang-format on
{
    if (dim == 0) return {nullptr, 0};
    auto const     count = detail::round_up_to<VectorSize>(dim);
    constexpr auto max_size =
        std::numeric_limits<std::size_t>::max() / sizeof(T);
    if (count > max_size) {
        integer_overflow_in_allocation<T>(dim, count, max_size);
    }
    static_assert(VectorSize * sizeof(T) % Alignment == 0);
    auto* raw =
        reinterpret_cast<T*>(std::aligned_alloc(Alignment, count * sizeof(T)));
    if (raw == nullptr) { throw_with_trace(std::bad_alloc{}); }
    std::unique_ptr<T[], FreeDeleter> pointer{raw};

    // TODO(twesterhout): Measure whether this matters and should be removed.
    for (std::size_t j = dim; j < count; ++j) {
        ::new (raw + j) T;
    }

    return {std::move(pointer), count};
}

// clang-format off
template <class T, std::size_t Alignment, std::size_t VectorSize>
TCM_SWARM_FORCEINLINE
auto allocate_aligned_buffer(std::size_t const dim1, std::size_t const dim2)
    -> std::tuple<std::unique_ptr<T[], FreeDeleter>, std::size_t, std::size_t>
// clang-format on
{
    if (dim1 == 0 || dim2 == 0) return {nullptr, 0, 0};
    auto const stride = detail::round_up_to<VectorSize>(dim2);
    constexpr auto max_size =
        std::numeric_limits<std::size_t>::max() / sizeof(T);
    if (stride > max_size / dim1) {
        // TODO(twesterhout): Strictly speaking, these multiplications may
        // result in overflow, so the log message will be incorrect.
        integer_overflow_in_allocation<T>(dim1 * dim2, dim1 * stride, max_size);
    }
    static_assert(VectorSize * sizeof(T) % Alignment == 0);
    auto* raw = reinterpret_cast<T*>(
        std::aligned_alloc(Alignment, dim1 * stride * sizeof(T)));
    if (raw == nullptr) { throw_with_trace(std::bad_alloc{}); }
    std::unique_ptr<T[], FreeDeleter> pointer{raw};

    // TODO(twesterhout): Measure whether this matters and should be removed.
    for (std::size_t i = 0; i < dim1; ++i) {
        for (std::size_t j = dim2; j < stride; ++j) {
            ::new (raw + i * stride + j) T;
        }
    }

    return {std::move(pointer), dim1, stride};
}
} // namespace detail

/// \brief Allocates a new buffer for a vector of dimension `dim`.
///
/// The returned vector has size of ``dim`` rounded up to a multiple of
/// ``Vc::simd<T, Abi>::size()`` and is aligned to at least
/// ``Vc::memory_alignment_v<Vc::simd<T, Abi>>``.
template <class T, class Abi, class... Size_t>
auto allocate_aligned_buffer(Size_t... dimensions)
{
    static_assert(Vc::is_abi_tag_v<Abi>, "Abi must be a valid ABI tag.");
    if constexpr (detail::is_complex_v<T>) {
        using R = typename T::value_type;
        static_assert(std::is_floating_point_v<R>,
            "Only std::complex<R> where R is a floating point is supported.");
        constexpr auto vector_size = Vc::simd<R, Abi>::size();
        constexpr auto alignment   = detail::alignment<R, Abi>();
        return detail::allocate_aligned_buffer<T, alignment, vector_size>(dimensions...);
    }
    else {
        static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
            "Only floating point and integral types are supported.");
        constexpr auto vector_size = Vc::simd<T, Abi>::size();
        constexpr auto alignment   = detail::alignment<T, Abi>();
        return detail::allocate_aligned_buffer<T, alignment, vector_size>(dimensions...);
    }
}

TCM_SWARM_END_NAMESPACE

