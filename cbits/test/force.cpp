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

#include "../heisenberg.hpp"
#include "../monte_carlo.hpp"
#include "../spin_state.hpp"
#include "parser.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>

#define EXPECT_CFLOAT_NEAR(val1, val2, abs_error)                         \
    EXPECT_NEAR(val1.real(), val2.real(), abs_error);                     \
    EXPECT_NEAR(val1.imag(), val2.imag(), abs_error);

namespace fs = std::experimental::filesystem;

namespace {
auto force_calculation(std::string const& input_file,
    std::string const& output_file, int const number_runs,
    int const steps_per_spin, float const rel_error)
{
    using tcm::Rbm;
    using C           = Rbm::C;
    using R           = Rbm::R;

    auto const exists = [](auto const& s) { return fs::exists(s); };
    ASSERT_PRED1(exists, input_file);
    ASSERT_PRED1(exists, output_file);

    std::ifstream in{input_file};
    ASSERT_TRUE(in);
    auto const rbm = parse_rbm_input(in);
    in.close();

    std::ifstream out{output_file};
    ASSERT_TRUE(out);
    auto energies_expected = parse_force_input(out, rbm);
    out.close();

    try {
        for (auto const& expected : energies_expected) {
            auto const& [magnetisation, energy_expected, force_expected] = expected;

            tcm::MetropolisConfig config;
            int const num_threads[] = {4, 1, 1};
            config.runs(number_runs)
                .steps(tcm::Steps{1000,
                    static_cast<int>(steps_per_spin * rbm.size_visible()),
                    rbm.size_visible()})
                .threads(num_threads)
                .flips(2)
                .magnetisation(magnetisation)
                .restarts(5);
            C moments[4] = {};

            // Allocate space for energies
            auto const number_steps = number_runs * config.steps().count();
            auto force_buffer = std::get<0>(Rbm::allocate_buffer(rbm.size()));
            auto [gradients_buffer, _ignored_, gradients_ldim] =
                Rbm::allocate_buffer(number_steps, rbm.size());

            auto force = gsl::make_span(force_buffer.get(), rbm.size());
            auto gradients = tcm::Gradients<C>{tcm_Matrix{
                gradients_buffer.get(), static_cast<int>(number_steps),
                static_cast<int>(rbm.size()),
                static_cast<int>(gradients_ldim)}};

            tcm::sample_gradients(rbm,
                tcm::heisenberg_1D(rbm.size_visible(), true, 5.0), config,
                gsl::make_span(moments), force, gradients);

            std::cout << "=> M = " << magnetisation << ", E = " << moments[0]
                      << ", #steps = " << number_steps << '\n';

            for (auto i = 0; i < force.size(); ++i) {
                EXPECT_CFLOAT_NEAR(force[i],
                    force_expected[gsl::narrow_cast<std::size_t>(i)],
                    std::abs(force_expected[gsl::narrow_cast<std::size_t>(i)])
                        * rel_error);
            }
#if 0
            vector_type covariance_data(rbm.size() * rbm.size());
            Expects(energies.size() >= rbm.size());
            auto S = gsl::as_multi_span(covariance.data(), gsl::dim(rbm.size()),
                    gsl::dim(rbm.size()));
            tcm::covariance_matrix(
                gradients, S, gsl::make_span(workspace.data(), rbm.size()));

            std::cout << "=> M = " << magnetisation << ", F = ";
            std::copy(std::begin(force), std::end(force),
                std::ostream_iterator<C>{std::cout, ", "});
            std::cout << '\n';

            std::cout << "S = \n";
            for (auto i = 0; i < S.template extent<0>(); ++i) {
                for (auto j = 0; j < S.template extent<1>(); ++j) {
                    std::cout << S[{i, j}] << ", ";
                }
                std::cout << '\n';
            }
#endif
        }
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << '\n';
        auto const* st = boost::get_error_info<tcm::errinfo_backtrace>(e);
        if (st != nullptr) {
            std::cerr << "Backtrace:\n" << *st << '\n';
        }
        throw;
    }
}
} // unnamed namespace

#define TEST_FORCE_CALCULATION(n, m, i, runs, steps, err)                      \
    TEST(Rbm##n##x##m, Force##i##steps)                                        \
    {                                                                          \
        force_calculation("input/rbm_" #n "_" #m "_" #i ".in",                 \
            "input/force_" #n "_" #m "_" #i ".out", runs, steps, err);         \
    }

TEST_FORCE_CALCULATION(6, 6, 0, 8, 5000, 0.15)
// TEST_FORCE_CALCULATION(6, 6, 1, 4, 50000, 0.15)
// TEST_FORCE_CALCULATION(6, 6, 2, 4, 50000, 0.15)

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


