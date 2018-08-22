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

#ifndef TCM_SWARM_MCMC_CONFIG_IPP
#define TCM_SWARM_MCMC_CONFIG_IPP

#include "mcmc_config.hpp"
#include "detail/errors.hpp"
#include <gsl/span>
#include <optional>
#include <sstream>

TCM_SWARM_BEGIN_NAMESPACE

[[noreturn]] inline auto Steps::this_is_nonsense() -> void
{
    std::ostringstream msg;
    msg << *this << " makes no sense.";
    throw_with_trace(std::domain_error{msg.str()});
}

inline auto operator<<(std::ostream& out, Steps const& x) -> std::ostream&
{
    return out << "Steps {start = " << x.start() << ", stop = " << x.stop()
               << ", step = " << x.step() << "}";
}

constexpr Steps::Steps(index_type const stop) : _start{0}, _stop{stop}, _step{1}
{
    if (stop < 0) { this_is_nonsense(); }
}

constexpr Steps::Steps(
    index_type const start, index_type const stop, index_type const step)
    : _start{start}, _stop{stop}, _step{step}
{
    if (start < 0 || stop < 0 || step <= 0 || start > stop) {
        this_is_nonsense();
    }
}

constexpr Steps::Steps(gsl::span<index_type const, 3> const steps)
    : Steps{steps[0], steps[1], steps[2]}
{
}

TCM_SWARM_FORCEINLINE
decltype(auto) MetropolisConfig::steps(Steps s) noexcept
{
    _steps = s;
    return *this;
}

TCM_SWARM_FORCEINLINE
decltype(auto) MetropolisConfig::threads(gsl::span<int const, 3> const num_threads)
{
    using std::begin, std::end;
    if (std::any_of(begin(num_threads), end(num_threads),
            [](auto x) { return x < 0; })) {
        std::ostringstream msg;
        msg << "Thread configuration {" << num_threads[0] << ", "
            << num_threads[1] << ", " << num_threads[2] << "} is invalid.";
        throw_with_trace(std::runtime_error{msg.str()});
    }
    std::copy(begin(num_threads), end(num_threads), _n_threads);
    return *this;
}

TCM_SWARM_FORCEINLINE
decltype(auto) MetropolisConfig::runs(int const n)
{
    if (n < 0) {
        std::ostringstream msg;
        msg << "A number of runs which is smaller than zero (" << n
            << ") makes little sense.";
        throw_with_trace(std::runtime_error{msg.str()});
    }
    _n_runs = n;
    return *this;
}

TCM_SWARM_FORCEINLINE
decltype(auto) MetropolisConfig::flips(int const n)
{
    if (n < 1) {
        std::ostringstream msg;
        msg << "A number of spin-flips which is smaller than one (" << n
            << ") makes little sense.";
        throw_with_trace(std::runtime_error{msg.str()});
    }
    if (magnetisation().has_value() && n % 2 != 0) {
        std::ostringstream msg;
        msg << "Cannot preserve magnetisation with an odd number of spin-"
               "flips ("
            << n << "). Try setting magnetisation to std::nullopt first.";
        throw_with_trace(std::runtime_error{msg.str()});
    }
    _n_flips = n;
    return *this;
}

TCM_SWARM_FORCEINLINE
decltype(auto) MetropolisConfig::restarts(int const n)
{
    if (n < 0) {
        std::ostringstream msg;
        msg << "Number of restarts cannot be negative (" << n << ").";
        throw_with_trace(std::runtime_error{msg.str()});
    }
    _max_restarts = n;
    return *this;
}

TCM_SWARM_FORCEINLINE
decltype(auto) MetropolisConfig::magnetisation(std::optional<int> const m)
{
    if (m.has_value() && flips() % 2 != 0) {
        std::ostringstream msg;
        msg << "Cannot preserve magnetisation with an odd number of spin-"
               "flips ("
            << *m
            << "). Try setting the number of spin-flips to an even natural "
               "number first.";
        throw_with_trace(std::runtime_error{msg.str()});
    }
    _magnetisation = m;
    return *this;
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_MCMC_CONFIG_IPP
