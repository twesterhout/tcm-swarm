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

// This ugly piece of code is only used for parsing test-files. Don't even think
// about using it in real life. It's not exception safe and probably just plain
// wrong in some places.

#ifndef TCM_SWARM_TEST_PARSER_HPP
#define TCM_SWARM_TEST_PARSER_HPP

#include "../detail/mkl_allocator.hpp"
#include "../rbm_spin_float.hpp"

#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

struct TCM_SWARM_SYMBOL_VISIBLE parse_error : std::runtime_error {
    parse_error(std::string message);
    parse_error(std::istream::int_type const expected,
        std::istream::int_type const got, std::string message = "");
};

template <class T>
struct PyComplex {
    std::complex<T> unpack;
};

template <class T>
struct PyList {
    std::vector<T> unpack;
};

struct PyRbm {
    tcm::Rbm unpack;
};

template <class T>
auto operator>>(std::istream& is, PyComplex<T>& x) -> std::istream&;

template <class T>
auto operator>>(std::istream& is, PyList<T>& x) -> std::istream&;

auto operator>>(std::istream& is, PyRbm& x) -> std::istream&;

TCM_SWARM_SYMBOL_IMPORT
auto parse_rbm_input(std::istream& is) -> tcm::Rbm;

TCM_SWARM_SYMBOL_IMPORT
auto parse_log_wf_input(std::istream& is, tcm::Rbm const& rbm)
    -> std::vector<std::pair<
        std::vector<tcm::Rbm::C, tcm::mkl::mkl_allocator<tcm::Rbm::C, 64>>,
        tcm::Rbm::C>>;

TCM_SWARM_SYMBOL_IMPORT
auto parse_energy_input(std::istream& is)
    -> std::vector<std::pair<tcm::Rbm::index_type, tcm::Rbm::C>>;

TCM_SWARM_SYMBOL_IMPORT
auto parse_der_log_wf_input(std::istream& is, tcm::Rbm const& rbm)
    -> std::vector<std::pair<
        std::vector<tcm::Rbm::C, tcm::mkl::mkl_allocator<tcm::Rbm::C, 64>>,
        std::vector<tcm::Rbm::C, tcm::mkl::mkl_allocator<tcm::Rbm::C, 64>>>>;

TCM_SWARM_SYMBOL_IMPORT
auto parse_force_input(std::istream& is, tcm::Rbm const& rbm)
    -> std::vector<std::tuple<tcm::Rbm::index_type, tcm::Rbm::C,
        std::vector<tcm::Rbm::C, tcm::mkl::mkl_allocator<tcm::Rbm::C, 64>>>>;

#endif // TCM_SWARM_TEST_PARSER_HPP
