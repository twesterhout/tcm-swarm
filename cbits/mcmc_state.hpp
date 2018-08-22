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

#ifndef TCM_SWARM_MCMC_STATE_HPP
#define TCM_SWARM_MCMC_STATE_HPP

#include "detail/config.hpp"

#include <any>
#include <complex>
#include <gsl/span>

TCM_SWARM_BEGIN_NAMESPACE

/// Abstract base class for Monte-Carlo states.
///
/// Monte-Carlo samplers should only rely on this interface.
class McmcState {
  public:
    using R          = float;
    using C          = std::complex<R>;
    using index_type = std::ptrdiff_t;
    using size_type  = index_type;

    constexpr McmcState() noexcept = default;

    McmcState(McmcState const&)     = delete;
    McmcState(McmcState&&) noexcept = delete;
    McmcState& operator=(McmcState const&) noexcept = delete;
    McmcState& operator=(McmcState&&) noexcept = delete;

    virtual ~McmcState() noexcept = default;

    /// Returns the number of visible units in the wrapped machine.
    auto size_visible() const noexcept -> size_type;

    /// Returns the total number of tunable parameters in the wrapped machine.
    auto size() const noexcept -> size_type;

    /// Returns :math:`\log\frac{\psi(\mathcal{S'})}{\psi(\mathcal{S})}` where
    /// :math:`\mathcal{S'}` is obtained from :math:`\mathcal{S}` by flipping
    /// spins at indices indicated by ``flips``. Additionally, a cache cell of
    /// unspecified type can be returned. This cell will then be passed to
    /// :cpp:func:`McmcState::update()` to speed up the calculation. If you do
    /// not wish to make use of caching, just use a default constructed
    /// ``std::any``.
    ///
    /// :param flips: An array of indices at which to flip spins. Each index
    /// must be in ``[0, size_visible() - 1)``. On top of that, indices must be
    /// unique!
    auto log_quot_wf(gsl::span<index_type const> flips) const -> std::tuple<C, std::any>;

    auto log_wf() const noexcept -> C;

    /// Calculates :math:`\partial_{\mathcal{W}_i} \log(\psi(\mathcal{S}))` and
    /// stores the result into ``out``.
    ///
    /// :param out: A contiguous vector of :cpp:func:`size()` complex numbers.
    auto der_log_wf(gsl::span<C> const out) const noexcept(!detail::gsl_can_throw()) -> void;

    /// Updates the current spin configuration by flipping spins at indices
    /// stored in ``flips``. Additionally, you can pass in a cache cell of
    /// unspecified type that was previously obtained from a call to
    /// :cpp:func:`log_quot_wf()`.
    ///
    /// :param flips: An array of indices at which to flip spins. Each index
    /// must be in ``[0, size_visible() - 1)``. On top of that, indices must be
    /// unique!
    auto update(gsl::span<index_type const> flips, std::any const& cache) -> void;

    /// Resets the state to the given visible units configuration.
    auto spin() const noexcept -> gsl::span<C const>;

    /// Resets the state to the given visible units configuration.
    auto spin(gsl::span<C const> spin) noexcept(!detail::gsl_can_throw()) -> void;

  private:
    virtual auto do_size_visible() const noexcept -> size_type = 0;
    virtual auto do_size() const noexcept -> size_type = 0;
    virtual auto do_log_quot_wf(gsl::span<index_type const> flips) const -> std::tuple<C, std::any> = 0;
    virtual auto do_log_wf() const noexcept -> C = 0;
    virtual auto do_der_log_wf(gsl::span<C> const out) const noexcept(!detail::gsl_can_throw()) -> void = 0;
    virtual auto do_update(gsl::span<index_type const> flips, std::any const& cache) -> void = 0;
    virtual auto do_spin(gsl::span<C const> spin) noexcept(!detail::gsl_can_throw()) -> void = 0;
    virtual auto do_spin() const noexcept -> gsl::span<C const> = 0;
};

TCM_SWARM_END_NAMESPACE

// And the implementation
#include "mcmc_state.ipp"

#endif // TCM_SWARM_MCMC_STATE_HPP
