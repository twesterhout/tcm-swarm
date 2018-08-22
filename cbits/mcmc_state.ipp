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

#ifndef TCM_SWARM_MCMC_STATE_IPP
#define TCM_SWARM_MCMC_STATE_IPP

#include "mcmc_state.hpp"

#include <any>
#include <tuple>
#include <vector>

TCM_SWARM_BEGIN_NAMESPACE

TCM_SWARM_FORCEINLINE
auto McmcState::size_visible() const noexcept -> size_type
{
    return do_size_visible();
}

TCM_SWARM_FORCEINLINE
auto McmcState::size() const noexcept -> size_type
{
    return do_size();
}

namespace detail {
// clang-format off
template <class IndexType, std::ptrdiff_t Extent>
TCM_SWARM_FORCEINLINE
auto flips_are_within_bounds(gsl::span<IndexType, Extent> const flips,
    std::remove_cv_t<IndexType> const  n) noexcept(!detail::gsl_can_throw())
    -> bool
// clang-format on
{
    using std::cbegin, std::cend;
    return std::all_of(cbegin(flips), cend(flips),
        [n](auto const i) { return 0 <= i && i < n; });
}

// clang-format off
template <class IndexType, std::ptrdiff_t Extent>
TCM_SWARM_FORCEINLINE
auto flips_are_unique(gsl::span<IndexType, Extent> const flips) -> bool
// clang-format on
{
    using std::begin, std::end;
    using std::cbegin, std::cend;
    std::vector<std::remove_cv_t<IndexType>> temp_flips{
        cbegin(flips), cend(flips)};
    std::sort(begin(temp_flips), end(temp_flips));
    return std::adjacent_find(begin(temp_flips), end(temp_flips))
           == end(temp_flips);
}
} // namespace detail


TCM_SWARM_FORCEINLINE
auto McmcState::log_wf() const noexcept -> C
{
    return do_log_wf();
}

TCM_SWARM_FORCEINLINE
auto McmcState::log_quot_wf(gsl::span<index_type const> const flips) const
    -> std::tuple<C, std::any>
{
    Expects(detail::flips_are_within_bounds(flips, size_visible()));
    Expects(detail::flips_are_unique(flips));
    return do_log_quot_wf(flips);
}

TCM_SWARM_FORCEINLINE
auto McmcState::der_log_wf(gsl::span<C> const out) const
    noexcept(!detail::gsl_can_throw()) -> void
{
    Expects(out.size() == size());
    do_der_log_wf(out);
}

TCM_SWARM_FORCEINLINE
auto McmcState::update(
    gsl::span<index_type const> const flips, std::any const& cache) -> void
{
    Expects(detail::flips_are_within_bounds(flips, size_visible()));
    Expects(detail::flips_are_unique(flips));
    do_update(flips, cache);
}

TCM_SWARM_FORCEINLINE
auto McmcState::spin(gsl::span<C const> const spin) noexcept(
    !detail::gsl_can_throw()) -> void
{
    Expects(spin.size() == size_visible());
    do_spin(spin);
}

TCM_SWARM_FORCEINLINE
auto McmcState::spin() const noexcept -> gsl::span<C const>
{
    return do_spin();
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_MCMC_STATE_IPP
