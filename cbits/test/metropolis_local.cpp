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
#include "../spin_state.hpp"
#include "../monte_carlo.hpp"
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
auto energy_calculation(std::string const& input_file,
    std::string const& output_file, int const number_runs,
    int const steps_per_spin, float const rel_error)
{
    using tcm::Rbm;
    using C = Rbm::C;
    using R = Rbm::R;
    auto const exists = [](auto const& s) { return fs::exists(s); };
    ASSERT_PRED1(exists, input_file);
    ASSERT_PRED1(exists, output_file);

    std::ifstream in{input_file};
    ASSERT_TRUE(in);
    auto const rbm = parse_rbm_input(in);
    in.close();

    std::ifstream out{output_file};
    ASSERT_TRUE(out);
    auto energies_expected = parse_energy_input(out);
    out.close();

    try {
        for (auto expected : energies_expected) {
            auto const [magnetisation, energy_expected] = expected;

            tcm::MetropolisConfig config;
            int const num_threads[] = {4, 1, 1};
            config.runs(number_runs)
                .steps(tcm::Steps{1000, static_cast<int>(steps_per_spin * rbm.size_visible()),
                    rbm.size_visible()})
                .threads(num_threads)
                .flips(2)
                .magnetisation(magnetisation)
                .restarts(5);
            C moments[1] = {};
            auto const [dim, _variance] = tcm::sample_moments(rbm,
                tcm::heisenberg_1D(rbm.size_visible(), true, 5.0, {1, 1}),
                config, gsl::make_span(moments));
            auto const variance = _variance.value();
            auto const energy = moments[0];
            EXPECT_NEAR(
                energy.real(), energy_expected.real(), 2 * std::sqrt(variance));
            EXPECT_NEAR(
                energy.imag(), energy_expected.imag(), 2 * std::sqrt(variance));
            EXPECT_CFLOAT_NEAR(energy, energy_expected,
                std::abs(energy_expected) * R{rel_error});
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

#define TEST_ENERGY_CALCULATION(n, m, i, runs, steps, err)                     \
    TEST(Rbm##n##x##m, Energy##i)                                              \
    {                                                                          \
        energy_calculation("input/rbm_" #n "_" #m "_" #i ".in",                \
            "input/heisenberg_" #n "_" #m "_" #i ".out", runs, steps, err);    \
    }

TEST_ENERGY_CALCULATION(6, 6, 0,  4, 4000, 0.1)
TEST_ENERGY_CALCULATION(6, 6, 19, 8, 4000, 0.1) // This one is extremely tricky
TEST_ENERGY_CALCULATION(6, 24, 0, 4, 4000, 0.1)
#if 0
TEST_ENERGY_CALCULATION(6, 6,  0, 4, 2000, 0.01)
TEST_ENERGY_CALCULATION(6, 6,  1, 4, 2500, 0.01)
TEST_ENERGY_CALCULATION(6, 6,  2, 4, 8000, 0.01)
TEST_ENERGY_CALCULATION(6, 6,  3, 4, 2500, 0.01)
TEST_ENERGY_CALCULATION(6, 6,  4, 4, 2500, 0.01)
TEST_ENERGY_CALCULATION(6, 6,  5, 4, 2000, 0.01)
TEST_ENERGY_CALCULATION(6, 6,  6, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6,  7, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6,  8, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6,  9, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 10, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 11, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 12, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 13, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 14, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 15, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 16, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 17, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 18, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 19, 4, 2000, 0.02)
TEST_ENERGY_CALCULATION(6, 6, 20, 4, 2000, 0.02)
// TEST_ENERGY_CALCULATION(6, 6, 19)
#endif

#if 1

#endif

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

