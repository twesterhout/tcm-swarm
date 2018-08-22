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

#ifndef TCM_SWARM_MCMC_CONFIG_HPP
#define TCM_SWARM_MCMC_CONFIG_HPP

#include "detail/config.hpp"
#include <gsl/span>
#include <optional>

TCM_SWARM_BEGIN_NAMESPACE

class Steps {
  public:
    using index_type = std::ptrdiff_t;

    constexpr Steps() noexcept : _start{0}, _stop{0}, _step{1} {}
    constexpr explicit Steps(index_type stop);
    constexpr Steps(index_type start, index_type stop, index_type step = 1);
    constexpr explicit Steps(gsl::span<index_type const, 3> steps);

    constexpr Steps(Steps const&) noexcept = default;
    constexpr Steps(Steps&&) noexcept      = default;
    constexpr Steps& operator=(Steps const&) noexcept = default;
    constexpr Steps& operator=(Steps&&) noexcept = default;

    constexpr auto start() const noexcept { return _start; }
    constexpr auto stop() const noexcept { return _stop; }
    constexpr auto step() const noexcept { return _step; }

    constexpr auto count() const noexcept -> index_type
    {
        return (_stop - _start - 1) / _step + 1;
    }

    constexpr auto copy_to(gsl::span<index_type, 3> const steps) const noexcept
        -> void
    {
        steps[0] = _start;
        steps[1] = _stop;
        steps[2] = _step;
    }

    friend auto operator<<(std::ostream& out, Steps const& x) -> std::ostream&;

  private:
    index_type _start;
    index_type _stop;
    index_type _step;

    [[noreturn]] auto this_is_nonsense() -> void;
};

class MetropolisConfig {
  public:
    /// Some sensible defaults.
    MetropolisConfig()
        : _steps{}
        , _n_threads{1, 1, 1}
        , _n_runs{}
        , _n_flips{1}
        , _max_restarts{}
        , _magnetisation{std::nullopt}
    {
    }

    constexpr auto const& steps() const noexcept { return _steps; }
    constexpr auto        threads() const noexcept
    {
        return gsl::make_span<int const, 3>(_n_threads);
    }
    constexpr auto        runs() const noexcept { return _n_runs; }
    constexpr auto        flips() const noexcept { return _n_flips; }
    constexpr auto        restarts() const noexcept { return _max_restarts; }
    constexpr auto const& magnetisation() const noexcept
    {
        return _magnetisation;
    }

    decltype(auto) steps(Steps s) noexcept;
    decltype(auto) threads(gsl::span<int const, 3>);
    decltype(auto) runs(int);
    decltype(auto) flips(int);
    decltype(auto) restarts(int);
    decltype(auto) magnetisation(std::optional<int> m);

    /*
    MetropolisConfig(tcm_MC_Config const& config)
    {
        this->steps(Steps{gsl::span{config.range}})
            .runs(config.runs)
            .flips(config.flips)
            .restarts(config.restarts)
            .magnetisation(config.has_magnetisation
                               ? std::optional{config.magnetisation}
                               : std::nullopt);
    }
    */

  private:
    Steps              _steps;
    int                _n_threads[3];
    int                _n_runs;
    int                _n_flips;
    int                _max_restarts;
    std::optional<int> _magnetisation;
};

TCM_SWARM_END_NAMESPACE

// And the implementation
#include "mcmc_config.ipp"

#endif // TCM_SWARM_MCMC_CONFIG_HPP
