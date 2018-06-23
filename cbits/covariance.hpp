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

#ifndef TCM_SWARM_COVARIANCE_HPP
#define TCM_SWARM_COVARIANCE_HPP

#include "detail/config.hpp"
#include "detail/herk.hpp"
#include "detail/her.hpp"
#include "detail/axpy.hpp"
#include "detail/scale.hpp"

#include <gsl/span>
#include <gsl/multi_span>

TCM_SWARM_BEGIN_NAMESPACE

namespace detail {

// TODO: This is a terribly inefficient implementation!
template <class C, std::ptrdiff_t Steps, std::ptrdiff_t Parameters>
auto average_derivative(
    gsl::multi_span<C const, Steps, Parameters> const derivatives,
    gsl::span<C, Parameters> const                    out)
{
    using std::begin, std::end;
    using R                      = typename C::value_type;
    auto const number_parameters = derivatives.template extent<1>();
    auto const number_steps      = derivatives.template extent<0>();
    Expects(out.size() == number_parameters);
    Expects(number_steps > 0);

    auto const first = derivatives[0];
    std::copy(begin(first), end(first), begin(out));
    for (auto i = 1; i < number_steps; ++i) {
        mkl::axpy(C{1}, derivatives[i], out);
    }
    mkl::scale(C{1} / gsl::narrow<R>(number_steps), out);
}

} // namespace detail

template <class C, std::ptrdiff_t Steps, std::ptrdiff_t Parameters>
auto covariance_matrix(
    gsl::multi_span<C const, Steps, Parameters> const derivatives,
    gsl::multi_span<C, Parameters, Parameters> const  out,
    gsl::span<C, Parameters> const                    workspace)
{
    using R = typename C::value_type;
    using std::begin, std::end;
    auto const number_parameters = derivatives.template extent<1>();
    auto const number_steps      = derivatives.template extent<0>();
    Expects(number_parameters > 0 && number_steps > 0);
    Expects(workspace.size() == number_parameters);
    Expects(out.template extent<0>() == number_parameters
            && out.template extent<1>() == number_parameters);

    mkl::herk(mkl::Layout::RowMajor, mkl::UpLo::Upper,
        mkl::Transpose::ConjTrans, R{1} / gsl::narrow<R>(number_steps),
        derivatives, R{0}, out);
    detail::average_derivative(derivatives, workspace);
    std::transform(begin(workspace), end(workspace), begin(workspace),
        [](auto const z) noexcept { return std::conj(z); });
    mkl::her(mkl::Layout::RowMajor, mkl::UpLo::Upper, R{-1},
        gsl::span<C const, Parameters>{workspace}, out);
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_COVARIANCE_HPP

