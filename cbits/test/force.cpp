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

#include <fstream>
#include <iostream>
#include <complex>
#include <experimental/filesystem>

#include <gtest/gtest.h>

#include "../heisenberg.hpp"
#include "../monte_carlo.hpp"
#include "../force.hpp"
#include "../covariance.hpp"
#include "../rbm_spin.hpp"

#include "../parse_test.hpp"

#define EXPECT_CFLOAT_NEAR(val1, val2, abs_error)                         \
    EXPECT_NEAR(val1.real(), val2.real(), abs_error);                     \
    EXPECT_NEAR(val1.imag(), val2.imag(), abs_error);

namespace fs = std::experimental::filesystem;

template <class Rbm>
auto force_calculation(std::string const& input_file,
    std::string const& output_file, long const number_runs,
    long const steps_per_spin, float const rel_error)
{
    static_assert(std::is_same_v<std::decay_t<Rbm>, Rbm>);
    using C           = typename Rbm::value_type;
    using R           = typename C::value_type;
    using vector_type = typename Rbm::vector_type;
    auto const exists = [](auto const& s) { return fs::exists(s); };
    ASSERT_PRED1(exists, input_file);
    ASSERT_PRED1(exists, output_file);

    std::ifstream in{input_file};
    ASSERT_TRUE(in);
    auto const rbm = parse_rbm_input<Rbm>(in);
    in.close();

    std::ifstream out{output_file};
    ASSERT_TRUE(out);
    auto energies_expected = parse_force_input<Rbm>(out, rbm);

    try {
        for (auto const& expected : energies_expected) {
            auto const& [magnetisation, energy_expected, force_expected] = expected;
            auto hamiltonian    = tcm::Heisenberg<R, 1, true>{5.0};
            auto steps          = std::make_tuple(1000l, rbm.size_visible(),
                1000l + steps_per_spin * rbm.size_visible());
            auto [energy, es, gs] = tcm::sample_gradients(rbm,
                std::move(hamiltonian), number_runs, std::move(steps), 2l,
                std::optional{magnetisation}, 5l);

            std::cout << "=> M = " << magnetisation << ", E = " << energy
                      << ", #steps = " << es.size() << '\n';

            auto energies  = gsl::span<C const>{es};
            auto gradients = gsl::as_multi_span<C const>(
                gs.data(), gsl::dim(es.size()), gsl::dim(rbm.size()));

            vector_type workspace_data(es.size());
            vector_type force_data(rbm.size());
            auto force = gsl::make_span(force_data);
            auto workspace = gsl::make_span(workspace_data);
            tcm::force(energies, energy, gradients, force, workspace);

            for (auto i = 0; i < force.size(); ++i) {
                EXPECT_CFLOAT_NEAR(force[i], force_expected[i],
                    std::abs(force_expected[i]) * rel_error);
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
        auto const* st = boost::get_error_info<tcm::traced>(e);
        if (st != nullptr) {
            std::cerr << "Backtrace:\n" << *st << '\n';
        }
        throw;
    }
}

#define TEST_FORCE_CALCULATION(n, m, i, runs, steps, err)                 \
    TEST(Rbm##n##x##m, Force##i##steps)                                   \
    {                                                                     \
        force_calculation<tcm::RbmBase<std::complex<float>>>(             \
            "input/rbm_" #n "_" #m "_" #i ".in",                          \
            "input/force_" #n "_" #m "_" #i ".out", runs, steps, err);    \
    }

TEST_FORCE_CALCULATION(6, 6, 0, 4, 5000, 0.15)
TEST_FORCE_CALCULATION(6, 6, 1, 4, 5000, 0.15)
TEST_FORCE_CALCULATION(6, 6, 2, 4, 10000, 0.15)

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


