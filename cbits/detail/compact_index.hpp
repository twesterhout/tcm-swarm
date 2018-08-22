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

#ifndef TCM_SWARM_DETAIL_COMPACT_INDEX_HPP
#define TCM_SWARM_DETAIL_COMPACT_INDEX_HPP

#include "errors.hpp"
#include <gsl/span>

TCM_SWARM_BEGIN_NAMESPACE
namespace detail {

/// Represents a (possibly known at compile-time) natural number.
/// This class is very similar to gsl::detail::extent_type.
template <class IndexType, std::ptrdiff_t Extent>
class CompactIndex;

// The value _is_ known at compile-time
template <class IndexType, std::ptrdiff_t Extent>
class CompactIndex {
  public:
    using index_type = IndexType;

    static_assert(Extent >= 0, "CompactIndex represents a natural number.");
    static_assert(std::is_convertible_v<std::ptrdiff_t, index_type>,
        "Invalid underlying type.");

    static constexpr auto extent() noexcept -> std::ptrdiff_t { return Extent; }

    constexpr auto index() const noexcept -> index_type
    {
        return gsl::narrow_cast<index_type>(Extent);
    }

    constexpr CompactIndex() noexcept                    = default;
    constexpr CompactIndex(CompactIndex const&) noexcept = default;
    constexpr CompactIndex(CompactIndex&&) noexcept      = default;
    constexpr CompactIndex& operator=(CompactIndex const&) noexcept = default;
    constexpr CompactIndex& operator=(CompactIndex&&) noexcept = default;

    constexpr CompactIndex(CompactIndex<index_type, gsl::dynamic_extent>
            other) noexcept(!detail::gsl_can_throw())
    {
        Expects(index() == other.index());
    }

    constexpr CompactIndex(index_type const other) noexcept(
        !detail::gsl_can_throw())
    {
        Expects(index() == other);
    }
};

// Value is not known until run-time
template <class IndexType>
class CompactIndex<IndexType, gsl::dynamic_extent> {
  public:
    using index_type = IndexType;

    static constexpr auto extent() noexcept -> std::ptrdiff_t
    {
        return gsl::dynamic_extent;
    }

    constexpr auto index() const noexcept -> index_type
    {
        return _index;
    }

    constexpr CompactIndex() noexcept                    = default;
    constexpr CompactIndex(CompactIndex const&) noexcept = default;
    constexpr CompactIndex(CompactIndex&&) noexcept      = default;
    constexpr CompactIndex& operator=(CompactIndex const&) noexcept = default;
    constexpr CompactIndex& operator=(CompactIndex&&) noexcept = default;

    template <std::ptrdiff_t Other>
    constexpr CompactIndex(CompactIndex<index_type, Other> const other) noexcept
        : _index{other.index()}
    {
        static_assert(
            Other == gsl::dynamic_extent || Other >= 0, "Invalid index.");
    }

    constexpr CompactIndex(index_type const other) noexcept : _index{other} {}

  private:
    index_type _index;
};

// The following explicit instantiations are mainly for testing.
template class CompactIndex<int, gsl::dynamic_extent>;
template class CompactIndex<int, 0>;
template class CompactIndex<int, 1>;
template class CompactIndex<int, 2>;
template class CompactIndex<int, 3>;
template class CompactIndex<int, 4>;
template class CompactIndex<int, 5>;

template class CompactIndex<long, gsl::dynamic_extent>;
template class CompactIndex<long, 0>;
template class CompactIndex<long, 1>;
template class CompactIndex<long, 2>;
template class CompactIndex<long, 3>;
template class CompactIndex<long, 4>;
template class CompactIndex<long, 5>;

} // namespace detail
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_COMPACT_INDEX_HPP
