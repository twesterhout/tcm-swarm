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


#ifndef TCM_SWARM_DETAIL_MKL_ALLOCATOR_HPP
#define TCM_SWARM_DETAIL_MKL_ALLOCATOR_HPP

#include <new>
#include <stdexcept>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

namespace detail {
namespace {
    constexpr auto is_valid_alignment(std::size_t const n) noexcept
        -> bool
    {
        return (n > 0) && ((n & (n - 1)) == 0);
    }
}
} // namespace detail

template <class T, std::size_t Alignment = 64>
class mkl_allocator {

    static_assert(detail::is_valid_alignment(Alignment),
        "Invalid `Alignment` argument. `Alignment` must be a "
        "non-negative power of 2.");

  public:
    // Public member types from "The C++17 Standard" (17.5.3.5,
    // "Allocator requirements")

    using pointer         = T*;
    using const_pointer   = T const*;
    using reference       = T&;
    using const_reference = T const&;
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using is_always_equal = std::true_type;

    // Construction and destruction.

    // clang-format off
    constexpr mkl_allocator()
        noexcept = default;

    constexpr mkl_allocator(mkl_allocator const& /*unused*/)
        noexcept = default;

    constexpr mkl_allocator(mkl_allocator && /*unused*/)
        noexcept = default;

    constexpr mkl_allocator& operator=(mkl_allocator const& /*unused*/)
        noexcept = default;

    constexpr mkl_allocator& operator=(mkl_allocator&& /*unused*/)
        noexcept = default;
    // clang-format on

    template <class U>
    explicit constexpr mkl_allocator(
        mkl_allocator<U> const& /*unused*/) noexcept
    {
    }

    template <class U>
    explicit constexpr mkl_allocator(
        mkl_allocator<U>&& /*unused*/) noexcept
    {
    }

    ~mkl_allocator() = default;

    // Public member functions

    /// \brief  Allocates memory for \p n objects of type #value_type.
    /// \note   Objects are not constructed.
    /// \return Pointer to the allocated memory. If `n == 0`,
    /// `nullptr` is
    ///         returned instead.
    auto allocate(size_type const n) const -> pointer
    {
        if (n == 0) {
            return nullptr;
        }
        if (n > max_size()) {
            throw std::length_error{"mkl_allocator<T>::allocate()"
                                    ": Integer overflow."};
        }
        auto* p = mkl_malloc(n * sizeof(value_type), Alignment);
        if (p == nullptr) {
            throw std::bad_alloc{};
        }
        return static_cast<pointer>(p);
    }

    /// \brief Synonym to #allocate, the hint parameter is not used by
    /// this
    ///        allocator.
    template <class U>
    auto allocate(size_type const n, U const* const /*unused*/) const
        -> pointer
    {
        return allocate(n);
    }

    /// \brief Deallocates memory previously allocated by a call to
    ///        #allocate.
    auto deallocate(pointer const p, size_type const /*unused*/) const
        noexcept
    {
        mkl_free(p);
    }

    constexpr auto address(reference x) const noexcept -> pointer
    {
        return std::addressof(x);
    }

    constexpr auto address(const_reference x) const noexcept
        -> const_pointer
    {
        return std::addressof(x);
    }

    /// \brief Calculates the largest value that can be passed to
    /// #allocate.
    constexpr auto max_size() const noexcept -> size_type
    {
        return std::numeric_limits<size_type>::max()
               / sizeof(value_type);
    }

    template <class U>
    struct rebind {
        using other = mkl_allocator<U>;
    };

    /// \brief Always returns `true`; this allocator is stateless.
    constexpr auto operator==(mkl_allocator const& /*unused*/) const
        noexcept -> bool
    {
        return true;
    }

    /// \brief Always returns `false`; this allocator is stateless.
    constexpr auto operator!=(mkl_allocator const& /*unused*/) const
        noexcept -> bool
    {
        return false;
    }

    /// \brief Constructs an object of #value_type at \p p.
    template <class... Args>
    auto construct(pointer const p, Args&&... args) const noexcept(
        std::is_nothrow_constructible<value_type, Args&&...>::value)
    {
        ::new (static_cast<void*>(p))
            value_type{std::forward<Args>(args)...};
    }

    /// \brief Destructs an object of #value_type at \p p.
    constexpr auto destroy(pointer const p) const noexcept
    {
        p->~T();
    }
};

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_MKL_ALLOCATOR_HPP
