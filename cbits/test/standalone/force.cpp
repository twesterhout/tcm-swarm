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

#if 0
#include <iostream>
#include <algorithm>
#include <iterator>
#include <complex>
#endif
#include "../../force.hpp"
#include "../../covariance.hpp"

int main()
{
#if 0
    using namespace std::complex_literals;
    std::complex<float> derivatives_data[] = {
        {0.3140814f,  0.95205253f}, {0.06467179f, 0.2858406f},  {0.28335367f, 0.0878391f},
        {0.02090741f, 0.10383991f}, {0.54635702f, 0.24284096f}, {0.13088325f, 0.31848011f},
        {0.04341057f, 0.42862717f}, {0.44661454f, 0.2049747f},  {0.62042639f, 0.17429626f},
        {0.3224134f, 0.21006665f},  {0.53341618f, 0.94049632f}, {0.52139325f, 0.20957638f}
    };
    std::complex<float> energies_data[] = {
        {0.73040935f, 0.67159879f},
        {0.71378849f, 0.18327124f},
        {0.01142997f, 0.74382297f},
        {0.32714006f, 0.45568905f}
    };

    std::complex<float> mean_derivative_data[3];
    std::fill(std::begin(mean_derivative_data),
        std::end(mean_derivative_data), std::complex{0.0f, 0.0f});

    auto const derivatives = gsl::as_multi_span<std::complex<float> const>(
        std::data(derivatives_data), gsl::dim<4>(), gsl::dim<3>());
    auto const energies = gsl::make_span<std::complex<float> const>(energies_data);
    auto const mean_derivative = gsl::make_span(mean_derivative_data);

    tcm::mkl::axpy(0.25f, gsl::make_span(&(derivatives[{0, 0}]), 3), mean_derivative);
    tcm::mkl::axpy(0.25f, gsl::make_span(&(derivatives[{1, 0}]), 3), mean_derivative);
    tcm::mkl::axpy(0.25f, gsl::make_span(&(derivatives[{2, 0}]), 3), mean_derivative);
    tcm::mkl::axpy(0.25f, gsl::make_span(&(derivatives[{3, 0}]), 3), mean_derivative);
    std::transform(std::begin(mean_derivative), std::end(mean_derivative),
        std::begin(mean_derivative), [](auto const x){ return std::conj(x); });

    std::copy(std::begin(mean_derivative), std::end(mean_derivative),
        std::ostream_iterator<std::complex<float>>{std::cout, ", "});
    std::cout << '\n';

    std::complex<float> force_data[3];
    std::complex<float> workspace_data[4];
    std::complex<float> covariance_data[9];

    auto const force = gsl::make_span(force_data);
    auto const workspace = gsl::make_span(workspace_data);
    auto const covariance =
        gsl::as_multi_span<std::complex<float>>(covariance_data, gsl::dim<3>(), gsl::dim<3>());

    tcm::force(energies, derivatives, force, workspace);

    std::copy(std::begin(force), std::end(force),
        std::ostream_iterator<std::complex<float>>{std::cout, ", "});
    std::cout << '\n';

    tcm::covariance_matrix(derivatives,
        gsl::span<std::complex<float> const, 3>{mean_derivative},
        covariance);

    std::copy(std::begin(covariance), std::end(covariance),
        std::ostream_iterator<std::complex<float>>{std::cout, ", "});
    std::cout << '\n';
    // Expected output:
    // (0.0601499,-0.0134551), (-0.0336545,-0.011075), (-0.0697014,0.0260818),
#endif
}
