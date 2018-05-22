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

#ifndef TCM_SWARM_FORCE_HPP
#define TCM_SWARM_FORCE_HPP

#include "detail/config.hpp"
#include "detail/axpy.hpp"
#include "detail/gemv.hpp"

#include <algorithm>
#include <numeric>

#include <gsl/span>
#include <gsl/multi_span>

TCM_SWARM_BEGIN_NAMESPACE

template <class C, std::ptrdiff_t Steps, std::ptrdiff_t Parameters>
auto force(gsl::span<C const, Steps> const            energies,
    gsl::multi_span<C const, Steps, Parameters> const derivatives,
    gsl::span<C, Parameters> const                    out,
    gsl::span<C, Steps> const                         workspace)
{
    auto const number_steps = energies.size();
    Expects(derivatives.template extent<0>() == number_steps);
    auto const number_parameters = derivatives.template extent<1>();
    Expects(out.size() == number_parameters);
    Expects(workspace.size() == number_steps);
    Expects(number_steps > 0 && number_parameters > 0);

    using R = typename C::value_type;
    // 〈E〉
    auto const mean_energy =
        std::accumulate(begin(energies), end(energies), C{0})
        / gsl::narrow<R>(energies.size());
    // workspace <- E - 〈E〉
    std::fill(begin(workspace), end(workspace), -mean_energy);
    mkl::axpy(C{1}, energies, workspace);
    // out <- 1/number_steps * O^H (E - 〈E〉)
    mkl::gemv(mkl::Layout::RowMajor, mkl::Transpose::ConjTrans,
        C{1} / gsl::narrow<R>(energies.size()), derivatives,
        gsl::span<C const, Steps>{workspace}, C{0}, out);
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_FORCE_HPP

