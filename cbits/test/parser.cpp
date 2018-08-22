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

#include "parser.hpp"

#include <iostream>
#include <complex>
#include <stdexcept>
#include <vector>

#include <gsl/gsl>

parse_error::parse_error(std::istream::int_type const expected,
    std::istream::int_type const got, std::string message)
    : std::runtime_error{"Parse error: expected '"
                         + std::string{static_cast<char>(expected)} + "' ("
                         + std::to_string(expected) + "), but got '"
                         + std::string{static_cast<char>(got)} + "' ("
                         + std::to_string(got) + "). " + message}
{
}

parse_error::parse_error(std::string const message)
    : std::runtime_error{message}
{
}

template <class CharT, class Traits>
auto& expects(std::basic_istream<CharT, Traits>& is, CharT const token)
{
    auto const ch = is.peek();
    if (token != ch) {
        is.setstate(std::ios_base::failbit);
        throw parse_error{token, ch};
    }
    is.get();
    return is;
}

template <class T>
auto operator>>(std::istream& is, PyComplex<T>& x) -> std::istream&
{
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        T a, b;
        std::ws(is);
        if (is.peek() != '(') {
            // We have a purely imaginary number
            a = 0;
            is >> b;
            expects(is, 'j');
            x.unpack = {a, b};
        }
        else {
            expects(is, '(');
            is >> a >> b;
            expects(is, 'j');
            std::ws(is);
            expects(is, ')');
            x.unpack = {a, b};
        }
    }
    catch (std::ios_base::failure const&) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse PyComplex<T>."};
    }
    return is;
}

template auto operator>>(std::istream&, PyComplex<float>&) -> std::istream&;
template auto operator>>(std::istream&, PyComplex<double>&) -> std::istream&;

template <class T>
auto operator>>(std::istream& is, PyList<T>& x) -> std::istream&
{
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        std::ws(is);
        expects(is, '[');
        if (x.unpack.size() > 0) {
            for (std::size_t i = 0; i < x.unpack.size() - 1; ++i) {
                is >> x.unpack[i];
                std::ws(is);
                expects(is, ',');
            }
            is >> x.unpack[x.unpack.size() - 1];
        }
        std::ws(is);
        expects(is, ']');
    }
    catch (std::ios_base::failure const&) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse PyList<T>."};
    }
    return is;
}

template auto operator>>(std::istream& is, PyList<PyComplex<float>>& x)
    -> std::istream&;
template auto operator>>(std::istream& is, PyList<PyComplex<double>>& x)
    -> std::istream&;
template auto operator>>(std::istream& is, PyList<PyList<PyComplex<float>>>& x)
    -> std::istream&;
template auto operator>>(std::istream& is, PyList<PyList<PyComplex<double>>>& x)
    -> std::istream&;

auto operator>>(std::istream& is, PyRbm& x) -> std::istream&
{
    using index_type  = tcm::Rbm::index_type;
    using C           = tcm::Rbm::C;
    using R           = tcm::Rbm::R;
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        auto& rbm = x.unpack;

        PyList<PyComplex<R>> a;
        a.unpack.resize(gsl::narrow_cast<std::size_t>(rbm.size_visible()));

        PyList<PyComplex<R>> b;
        b.unpack.resize(gsl::narrow_cast<std::size_t>(rbm.size_hidden()));

        PyList<PyList<PyComplex<R>>> w;
        w.unpack.resize(gsl::narrow_cast<std::size_t>(rbm.size_hidden()));
        for (auto& row : w.unpack) {
            row.unpack.resize(
                gsl::narrow_cast<std::size_t>(rbm.size_visible()));
        }

        is >> a >> b >> w;

        for (auto i = 0; i < rbm.size_visible(); ++i) {
            rbm.visible(i) = a.unpack[gsl::narrow_cast<std::size_t>(i)].unpack;
        }

        for (auto i = 0; i < rbm.size_hidden(); ++i) {
            rbm.hidden(i) = b.unpack[gsl::narrow_cast<std::size_t>(i)].unpack;
        }

        for (auto i = 0; i < rbm.size_hidden(); ++i) {
            for (auto j = 0; j < rbm.size_visible(); ++j) {
                rbm.weights(i, j) =
                    w.unpack[gsl::narrow_cast<std::size_t>(i)]
                        .unpack[gsl::narrow_cast<std::size_t>(j)]
                        .unpack;
            }
        }
    }
    catch (std::ios_base::failure const&) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse PyRbm<T>."};
    }
    return is;
}

TCM_SWARM_SYMBOL_EXPORT
auto parse_rbm_input(std::istream& is) -> tcm::Rbm
{
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        int n, m;
        is >> n >> m;
        PyRbm rbm{tcm::Rbm{n, m}};
        is >> rbm;
        return std::move(rbm.unpack);
    }
    catch (std::ios_base::failure const&) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse Rbm."};
    }
}

TCM_SWARM_SYMBOL_EXPORT
auto parse_log_wf_input(std::istream& is, tcm::Rbm const& rbm)
    -> std::vector<std::pair<
        std::vector<tcm::Rbm::C, tcm::mkl::mkl_allocator<tcm::Rbm::C, 64>>,
        tcm::Rbm::C>>
{
    using tcm::Rbm;
    using C             = Rbm::C;
    using R             = Rbm::R;
    using vector_type   = std::vector<C, tcm::mkl::mkl_allocator<C, 64>>;
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        std::vector<std::pair<vector_type, C>> expected;
        while (!is.eof()) {
            PyList<R>    s{std::vector<R>(
                gsl::narrow_cast<std::size_t>(rbm.size_visible()))};
            PyComplex<R> psi;
            is >> s >> psi;

            vector_type spin(std::begin(s.unpack), std::end(s.unpack));
            expected.emplace_back(std::move(spin), psi.unpack);
            std::ws(is);
        }
        return expected;
    }
    catch (std::ios_base::failure const&) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse spin - log_wf combinations."};
    }
}

TCM_SWARM_SYMBOL_EXPORT
auto parse_der_log_wf_input(std::istream& is, tcm::Rbm const& rbm)
    -> std::vector<std::pair<
        std::vector<tcm::Rbm::C, tcm::mkl::mkl_allocator<tcm::Rbm::C, 64>>,
        std::vector<tcm::Rbm::C, tcm::mkl::mkl_allocator<tcm::Rbm::C, 64>>>>
{
    using tcm::Rbm;
    using C           = Rbm::C;
    using R           = Rbm::R;
    using vector_type = std::vector<C, tcm::mkl::mkl_allocator<C, 64>>;
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        std::vector<std::pair<vector_type, vector_type>> expected;
        while (!is.eof()) {
            PyList<R>            py_spin{std::vector<R>(
                gsl::narrow_cast<std::size_t>(rbm.size_visible()))};
            PyList<PyComplex<R>> py_grad{std::vector<PyComplex<R>>(
                gsl::narrow_cast<std::size_t>(rbm.size()))};
            is >> py_spin;
            is >> py_grad;
            vector_type spin(
                std::begin(py_spin.unpack), std::end(py_spin.unpack));
            vector_type gradient;
            std::transform(std::begin(py_grad.unpack), std::end(py_grad.unpack),
                std::back_inserter(gradient), [](auto x) { return x.unpack; });
            expected.emplace_back(std::move(spin), std::move(gradient));
            std::ws(is);
        }
        return expected;
    }
    catch (std::ios_base::failure const&) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse spin - der_log_wf combinations."};
    }
}

TCM_SWARM_SYMBOL_EXPORT
auto parse_energy_input(std::istream& is)
    -> std::vector<std::pair<tcm::Rbm::index_type, tcm::Rbm::C>>
{
    using tcm::Rbm;
    using index_type = Rbm::index_type;
    using C          = Rbm::C;
    using R          = Rbm::R;
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        std::vector<std::pair<index_type, C>> expected;
        while (!is.eof()) {
            index_type   magnetisation;
            PyComplex<R> energy;
            is >> magnetisation >> energy;
            expected.emplace_back(magnetisation, energy.unpack);
            std::ws(is);
        }
        return expected;
    }
    catch (std::ios_base::failure const&) {
        is.exceptions(io_state);
        throw parse_error{
            "Failed to parse magnetisation - energy combinations."};
    }
}

TCM_SWARM_SYMBOL_EXPORT
auto parse_force_input(std::istream& is, tcm::Rbm const& rbm)
    -> std::vector<std::tuple<tcm::Rbm::index_type, tcm::Rbm::C,
        std::vector<tcm::Rbm::C, tcm::mkl::mkl_allocator<tcm::Rbm::C, 64>>>>
{
    using tcm::Rbm;
    using index_type    = Rbm::index_type;
    using C             = Rbm::C;
    using R             = Rbm::R;
    using vector_type   = std::vector<C, tcm::mkl::mkl_allocator<C, 64>>;
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        std::vector<std::tuple<index_type, C, vector_type>> expected;
        while (!is.eof()) {
            index_type   magnetisation;
            PyComplex<R> py_energy;
            PyList<PyComplex<R>> py_force{std::vector<PyComplex<R>>(
                gsl::narrow_cast<std::size_t>(rbm.size()))};
            is >> magnetisation >> py_energy >> py_force;
            vector_type force;
            std::transform(std::begin(py_force.unpack),
                std::end(py_force.unpack), std::back_inserter(force),
                [](auto x) { return x.unpack; });
            expected.emplace_back(
                magnetisation, py_energy.unpack, std::move(force));
            std::ws(is);
        }
        return expected;
    }
    catch (std::ios_base::failure const&) {
        is.exceptions(io_state);
        throw parse_error{
            "Failed to parse magnetisation - energy - force combinations."};
    }
}

