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

#include <gsl/span>
#include <gsl/multi_span>

TCM_SWARM_BEGIN_NAMESPACE

template <class C, std::ptrdiff_t Steps, std::ptrdiff_t Parameters>
auto covariance_matrix(
    gsl::multi_span<C const, Steps, Parameters> const derivatives,
    gsl::span<C const, Parameters> const              conj_mean_derivative,
    gsl::multi_span<C, Parameters, Parameters> const  out)
{
    using R = typename C::value_type;
    auto const number_parameters = derivatives.template extent<1>();
    auto const number_steps      = derivatives.template extent<0>();
    Expects(number_parameters > 0 && number_steps > 0);
    Expects(conj_mean_derivative.size() == number_parameters);
    Expects(out.template extent<0>() == number_parameters
            && out.template extent<1>() == number_parameters);

    mkl::herk(mkl::Layout::RowMajor, mkl::UpLo::Upper,
        mkl::Transpose::ConjTrans,
        R{1} / gsl::narrow<R>(number_steps), derivatives, R{0}, out);
#if 0
    std::copy(std::begin(out), std::end(out),
        std::ostream_iterator<C>{std::cout, ", "});
    std::cout << '\n';
#endif
    mkl::her(mkl::Layout::RowMajor, mkl::UpLo::Upper, R{-1},
        conj_mean_derivative, out);
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_COVARIANCE_HPP

