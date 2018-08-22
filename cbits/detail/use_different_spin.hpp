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

#ifndef TCM_SWARM_DETAIL_USE_DIFFERENT_SPIN_HPP
#define TCM_SWARM_DETAIL_USE_DIFFERENT_SPIN_HPP

#include "config.hpp"

#include <stdexcept>
#include <vector>

#include <gsl/span>

TCM_SWARM_BEGIN_NAMESPACE

struct use_different_spin : public virtual std::exception {
    // TODO(twesterhout): This should be taken from McmcState!
    using index_type = std::ptrdiff_t;

    template <std::size_t N>
    use_different_spin(std::array<index_type, N> flips) noexcept
        : _flips{std::begin(flips), std::end(flips)}
    {
    }

    use_different_spin(std::vector<index_type> flips) noexcept
        : _flips{std::move(flips)}
    {
    }

    virtual auto what() const noexcept -> char const* override
    {
        return "Try starting from a different spin configuration.";
    }

    auto flips() const & noexcept -> gsl::span<index_type const>
    {
        return {_flips};
    }

  private:
    std::vector<index_type> _flips;
};

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_USE_DIFFERENT_SPIN_HPP
