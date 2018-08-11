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
#include <complex>
#include <experimental/filesystem>

#include <gtest/gtest.h>

#include "../rbm_spin.hpp"
#include "../parse_test.hpp"

#define EXPECT_CFLOAT_NEAR(val1, val2, abs_error)                         \
    EXPECT_NEAR(val1.real(), val2.real(), abs_error);                     \
    EXPECT_NEAR(val1.imag(), val2.imag(), abs_error);

namespace fs = std::experimental::filesystem;

template <class IndexType, class C>
auto get_flips(
    gsl::span<C const> const spin_1, gsl::span<C const> const spin_2)
{
    Expects(spin_1.size() == spin_2.size());
    std::vector<IndexType> flips;
    for (IndexType i = 0; i < spin_1.size(); ++i) {
        if (spin_1[i] != spin_2[i]) {
            Expects(spin_1[i] == -spin_2[i]);
            flips.push_back(i);
        }
    }
    return flips;
}

template <class Rbm>
auto log_wf_calculation(
    std::string const& input_file, std::string const& output_file)
{
    static_assert(std::is_same_v<std::decay_t<Rbm>, Rbm>);
    using C = typename Rbm::value_type;
    using R = typename C::value_type;
    auto const exists = [](auto const& s) { return fs::exists(s); };
    ASSERT_PRED1(exists, input_file);
    ASSERT_PRED1(exists, output_file);

    std::ifstream in{input_file};
    ASSERT_TRUE(in);
    auto const rbm = parse_rbm_input<Rbm>(in);
    in.close();

    std::ifstream out{output_file};
    ASSERT_TRUE(out);
    auto spin_log_wf_expected = parse_log_wf_input(out, rbm);

    for (auto const& expected : spin_log_wf_expected) {
        auto [spin, log_wf_expected] = expected;
        auto const log_wf            = rbm.log_wf(gsl::span<C const>{spin});
        EXPECT_CFLOAT_NEAR(log_wf, log_wf_expected, R{5.00E-5});
    }
}

template <class Rbm>
auto log_wf_via_flips_calculation(
    std::string const& input_file, std::string const& output_file)
{
    static_assert(std::is_same_v<std::decay_t<Rbm>, Rbm>);
    using C = typename Rbm::value_type;
    using R = typename C::value_type;
    using index_type = typename Rbm::index_type;
    auto const exists = [](auto const& s) { return fs::exists(s); };
    ASSERT_PRED1(exists, input_file);
    ASSERT_PRED1(exists, output_file);

    std::ifstream in{input_file};
    ASSERT_TRUE(in);
    auto const rbm = parse_rbm_input<Rbm>(in);
    in.close();

    std::ifstream out{output_file};
    ASSERT_TRUE(out);
    auto spin_log_wf_expected = parse_log_wf_input(out, rbm);

    auto [initial_spin, initial_log_wf] = spin_log_wf_expected.front();
    tcm::McmcBase<C> state{rbm, std::move(initial_spin)};
    EXPECT_CFLOAT_NEAR(state.log_wf(), initial_log_wf,
        std::abs(initial_log_wf) * R{5.0E-5});
    for (auto const& expected : spin_log_wf_expected) {
        auto const& new_spin        = std::get<0>(expected);
        auto const  log_wf_expected = std::get<1>(expected);
        auto const  flips           = get_flips<index_type, C>(
            state.spin(), gsl::span<C const>{new_spin});
        auto const log_wf =
            std::get<0>(state.log_quot_wf(gsl::span<index_type const>{flips}))
            + state.log_wf();
        EXPECT_CFLOAT_NEAR(log_wf, log_wf_expected,
            std::abs(log_wf_expected) * R{5.0E-5});
    }
}

template <class Rbm>
auto der_log_wf_calculation(
    std::string const& input_file, std::string const& output_file)
{
    static_assert(std::is_same_v<std::decay_t<Rbm>, Rbm>);
    using C = typename Rbm::value_type;
    using R = typename C::value_type;
    auto const exists = [](auto const& s) { return fs::exists(s); };
    ASSERT_PRED1(exists, input_file);
    ASSERT_PRED1(exists, output_file);

    std::ifstream in{input_file};
    ASSERT_TRUE(in);
    auto const rbm = parse_rbm_input<Rbm>(in);
    in.close();

    std::ifstream out{output_file};
    ASSERT_TRUE(out);
    auto spin_der_log_wf_expected = parse_der_log_wf_input(out, rbm);

    Expects(spin_der_log_wf_expected.size() == 10);
    for (auto const& expected : spin_der_log_wf_expected) {
        auto const& [spin, der_log_wf_expected] = expected;
        auto const theta = rbm.theta(gsl::make_span(spin));
        std::vector<C> der_log_wf(rbm.size());
        rbm.der_log_wf(
            gsl::span<C const>{spin}, theta, gsl::make_span(der_log_wf));
#if 0
        std::cout << "O[0] = [";
        std::copy(begin(der_log_wf), end(der_log_wf),
            std::ostream_iterator<C>{std::cout, ", "});
        std::cout << "]\n";
        std::cout << "O[0]_expected = [";
        std::copy(begin(der_log_wf_expected), end(der_log_wf_expected),
            std::ostream_iterator<C>{std::cout, ", "});
        std::cout << "]\n";
#endif
        for (unsigned i = 0; i < der_log_wf.size(); ++i) {
            // std::cout << i << '\n';
            EXPECT_CFLOAT_NEAR(der_log_wf[i], der_log_wf_expected[i],
                std::abs(der_log_wf_expected[i]) * R{5.00E-4});
        }
    }
}

#define TEST_LOG_WF_CALCULATION(n, m, i)                                  \
    TEST(Rbm##n##x##m, LogWF##i)                                          \
    {                                                                     \
        log_wf_calculation<tcm::RbmBase<std::complex<float>>>(            \
            "input/rbm_" #n "_" #m "_" #i ".in",                          \
            "input/log_wf_" #n "_" #m "_" #i ".out");                     \
    }

#define TEST_LOG_WF_FLIPS(n, m, i)                                        \
    TEST(Rbm##n##x##m, LogWFviaFlips##i)                                  \
    {                                                                     \
        log_wf_via_flips_calculation<tcm::RbmBase<std::complex<float>>>(  \
            "input/rbm_" #n "_" #m "_" #i ".in",                          \
            "input/log_wf_" #n "_" #m "_" #i ".out");                     \
    }

#define TEST_DER_LOG_WF_CALCULATION(n, m, i)                              \
    TEST(Rbm##n##x##m, DerLogWF##i)                                       \
    {                                                                     \
        der_log_wf_calculation<tcm::RbmBase<std::complex<float>>>(        \
            "input/rbm_" #n "_" #m "_" #i ".in",                          \
            "input/gradient_" #n "_" #m "_" #i ".out");                   \
    }

TEST_LOG_WF_CALCULATION(6, 6, 0)
TEST_LOG_WF_CALCULATION(6, 6, 1)
TEST_LOG_WF_CALCULATION(6, 6, 2)
TEST_LOG_WF_CALCULATION(6, 6, 3)
TEST_LOG_WF_CALCULATION(6, 6, 4)
TEST_LOG_WF_CALCULATION(6, 6, 5)
TEST_LOG_WF_CALCULATION(6, 6, 6)
TEST_LOG_WF_CALCULATION(6, 6, 7)
TEST_LOG_WF_CALCULATION(6, 6, 8)
TEST_LOG_WF_CALCULATION(6, 6, 9)
TEST_LOG_WF_CALCULATION(6, 6, 10)
TEST_LOG_WF_CALCULATION(6, 6, 11)
// TEST_LOG_WF_CALCULATION(6, 6, 12)
TEST_LOG_WF_CALCULATION(6, 6, 13)
TEST_LOG_WF_CALCULATION(6, 6, 14)
TEST_LOG_WF_CALCULATION(6, 6, 15)
TEST_LOG_WF_CALCULATION(6, 6, 16)
TEST_LOG_WF_CALCULATION(6, 6, 17)
TEST_LOG_WF_CALCULATION(6, 6, 18)
TEST_LOG_WF_CALCULATION(6, 6, 19)
TEST_LOG_WF_CALCULATION(6, 24, 0)
TEST_LOG_WF_CALCULATION(6, 24, 1)
TEST_LOG_WF_CALCULATION(6, 24, 2)
TEST_LOG_WF_CALCULATION(6, 24, 3)
TEST_LOG_WF_CALCULATION(6, 24, 4)
TEST_LOG_WF_CALCULATION(6, 24, 5)
TEST_LOG_WF_CALCULATION(6, 24, 6)
TEST_LOG_WF_CALCULATION(6, 24, 7)
TEST_LOG_WF_CALCULATION(6, 24, 8)
TEST_LOG_WF_CALCULATION(6, 24, 9)
// TEST_LOG_WF_CALCULATION(6, 24, 10)
TEST_LOG_WF_CALCULATION(6, 24, 11)
TEST_LOG_WF_CALCULATION(6, 24, 12)
TEST_LOG_WF_CALCULATION(6, 24, 13)
TEST_LOG_WF_CALCULATION(6, 24, 14)
TEST_LOG_WF_CALCULATION(6, 24, 15)
TEST_LOG_WF_CALCULATION(6, 24, 16)
TEST_LOG_WF_CALCULATION(6, 24, 17)
TEST_LOG_WF_CALCULATION(6, 24, 18)
TEST_LOG_WF_CALCULATION(6, 24, 19)

TEST_LOG_WF_FLIPS(6, 6, 0)
TEST_LOG_WF_FLIPS(6, 6, 1)
TEST_LOG_WF_FLIPS(6, 6, 2)
TEST_LOG_WF_FLIPS(6, 6, 3)
TEST_LOG_WF_FLIPS(6, 6, 4)
TEST_LOG_WF_FLIPS(6, 6, 5)
TEST_LOG_WF_FLIPS(6, 6, 6)
TEST_LOG_WF_FLIPS(6, 6, 7)
TEST_LOG_WF_FLIPS(6, 6, 8)
TEST_LOG_WF_FLIPS(6, 6, 9)
TEST_LOG_WF_FLIPS(6, 6, 10)
TEST_LOG_WF_FLIPS(6, 6, 11)
TEST_LOG_WF_FLIPS(6, 6, 12)
TEST_LOG_WF_FLIPS(6, 6, 13)
TEST_LOG_WF_FLIPS(6, 6, 14)
TEST_LOG_WF_FLIPS(6, 6, 15)
TEST_LOG_WF_FLIPS(6, 6, 16)
TEST_LOG_WF_FLIPS(6, 6, 17)
TEST_LOG_WF_FLIPS(6, 6, 18)
TEST_LOG_WF_FLIPS(6, 6, 19)
TEST_LOG_WF_FLIPS(6, 24, 0)
TEST_LOG_WF_FLIPS(6, 24, 1)
TEST_LOG_WF_FLIPS(6, 24, 2)
TEST_LOG_WF_FLIPS(6, 24, 3)
TEST_LOG_WF_FLIPS(6, 24, 4)
TEST_LOG_WF_FLIPS(6, 24, 5)
TEST_LOG_WF_FLIPS(6, 24, 6)
TEST_LOG_WF_FLIPS(6, 24, 7)
TEST_LOG_WF_FLIPS(6, 24, 8)
TEST_LOG_WF_FLIPS(6, 24, 9)
TEST_LOG_WF_FLIPS(6, 24, 10)
TEST_LOG_WF_FLIPS(6, 24, 11)
TEST_LOG_WF_FLIPS(6, 24, 12)
TEST_LOG_WF_FLIPS(6, 24, 13)
TEST_LOG_WF_FLIPS(6, 24, 14)
TEST_LOG_WF_FLIPS(6, 24, 15)
TEST_LOG_WF_FLIPS(6, 24, 16)
TEST_LOG_WF_FLIPS(6, 24, 17)
TEST_LOG_WF_FLIPS(6, 24, 18)
TEST_LOG_WF_FLIPS(6, 24, 19)

TEST_DER_LOG_WF_CALCULATION(6, 6, 0)
TEST_DER_LOG_WF_CALCULATION(6, 6, 1)
TEST_DER_LOG_WF_CALCULATION(6, 6, 2)
TEST_DER_LOG_WF_CALCULATION(6, 6, 3)
TEST_DER_LOG_WF_CALCULATION(6, 6, 4)
TEST_DER_LOG_WF_CALCULATION(6, 6, 5)
TEST_DER_LOG_WF_CALCULATION(6, 6, 6)
TEST_DER_LOG_WF_CALCULATION(6, 6, 7)
TEST_DER_LOG_WF_CALCULATION(6, 6, 8)
TEST_DER_LOG_WF_CALCULATION(6, 6, 9)
TEST_DER_LOG_WF_CALCULATION(6, 6, 10)
TEST_DER_LOG_WF_CALCULATION(6, 6, 11)
TEST_DER_LOG_WF_CALCULATION(6, 6, 12)
TEST_DER_LOG_WF_CALCULATION(6, 6, 13)
TEST_DER_LOG_WF_CALCULATION(6, 6, 14)
TEST_DER_LOG_WF_CALCULATION(6, 6, 15)
TEST_DER_LOG_WF_CALCULATION(6, 6, 16)
TEST_DER_LOG_WF_CALCULATION(6, 6, 17)
TEST_DER_LOG_WF_CALCULATION(6, 6, 18)
TEST_DER_LOG_WF_CALCULATION(6, 6, 19)
TEST_DER_LOG_WF_CALCULATION(6, 24, 0)
TEST_DER_LOG_WF_CALCULATION(6, 24, 1)
TEST_DER_LOG_WF_CALCULATION(6, 24, 2)
TEST_DER_LOG_WF_CALCULATION(6, 24, 3)
TEST_DER_LOG_WF_CALCULATION(6, 24, 4)
TEST_DER_LOG_WF_CALCULATION(6, 24, 5)
TEST_DER_LOG_WF_CALCULATION(6, 24, 6)
TEST_DER_LOG_WF_CALCULATION(6, 24, 7)
TEST_DER_LOG_WF_CALCULATION(6, 24, 8)
TEST_DER_LOG_WF_CALCULATION(6, 24, 9)
TEST_DER_LOG_WF_CALCULATION(6, 24, 10)
TEST_DER_LOG_WF_CALCULATION(6, 24, 11)
TEST_DER_LOG_WF_CALCULATION(6, 24, 12)
TEST_DER_LOG_WF_CALCULATION(6, 24, 13)
TEST_DER_LOG_WF_CALCULATION(6, 24, 14)
TEST_DER_LOG_WF_CALCULATION(6, 24, 15)
TEST_DER_LOG_WF_CALCULATION(6, 24, 16)
TEST_DER_LOG_WF_CALCULATION(6, 24, 17)
TEST_DER_LOG_WF_CALCULATION(6, 24, 18)
TEST_DER_LOG_WF_CALCULATION(6, 24, 19)

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
