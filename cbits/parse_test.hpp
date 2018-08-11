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

#ifndef TCM_SWARM_PARSE_TEST_HPP
#define TCM_SWARM_PARSE_TEST_HPP

#include <iostream>
#include <complex>
#include <stdexcept>
#include <vector>

#include <gsl/gsl>

struct parse_error : std::runtime_error {
    parse_error(std::istream::int_type const expected,
        std::istream::int_type const got, std::string message = "")
        : std::runtime_error{
              "Parse error: expected '"
              + std::string{static_cast<char>(expected)} + "' ("
              + std::to_string(expected) + "), but got '"
              + std::string{static_cast<char>(got)} + "' ("
              + std::to_string(got) + "). " + message}
    {
    }

    parse_error(std::string message) : std::runtime_error{message} {}
};

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
struct PyComplex {
    std::complex<T> unpack;
};

template <class T, class CharT, class Traits>
auto& operator<<(
    std::basic_ostream<CharT, Traits>& os, PyComplex<T> const& x)
{
    return os << x.unpack;
}

template <class T, class CharT, class Traits>
auto operator>>(std::basic_istream<CharT, Traits>& is, PyComplex<T>& x)
    -> std::basic_istream<CharT, Traits>&
{
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        T a, b;
        std::ws(is);
        if (is.peek() != CharT{'('}) {
            // We have a purely imaginary number
            a = 0;
            is >> b;
            expects(is, CharT{'j'});
            x.unpack = {a, b};
        }
        else {
            expects(is, CharT{'('});
            is >> a >> b;
            expects(is, CharT{'j'});
            std::ws(is);
            expects(is, CharT{')'});
            x.unpack = {a, b};
        }
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse PyComplex<T>."};
    }
    return is;
}

template <class T, std::ptrdiff_t Extent = gsl::dynamic_extent>
struct PyList {
    gsl::span<T, Extent> unpack;
};

template <class T, std::ptrdiff_t Extent>
auto to_list(gsl::span<T, Extent> const x) -> PyList<T, Extent>
{
    return {x};
}

template <class T, std::ptrdiff_t Extent>
auto to_list(gsl::span<std::complex<T>, Extent> const x)
    -> PyList<PyComplex<T>, Extent>
{
    static_assert(sizeof(std::complex<T>) == sizeof(PyComplex<T>));
    return {gsl::span{reinterpret_cast<PyComplex<T>*>(x.data()), x.size()}};
}

template <class T, std::ptrdiff_t Extent>
auto to_list(gsl::span<std::complex<T> const, Extent> const x)
    -> PyList<PyComplex<T> const, Extent>
{
    static_assert(sizeof(std::complex<T>) == sizeof(PyComplex<T>));
    return {gsl::span{
        reinterpret_cast<PyComplex<T> const*>(x.data()), x.size()}};
}

template <class T, std::ptrdiff_t Extent, class CharT, class Traits>
auto operator>>(std::basic_istream<CharT, Traits>& is, PyList<T, Extent>& x)
    -> std::basic_istream<CharT, Traits>&
{
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        std::ws(is);
        expects(is, CharT{'['});
        if (x.unpack.size() > 0) {
            for (auto i = 0; i < x.unpack.size() - 1; ++i) {
                T el;
                is >> el;
                x.unpack[i] = std::move(el);
                std::ws(is);
                expects(is, CharT{','});
            }
            T el;
            is >> el;
            x.unpack[x.unpack.size() - 1] = std::move(el);
        }
        std::ws(is);
        expects(is, CharT{']'});
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse PyList<T, Extent>."};
    }
    return is;
}

template <class T, std::ptrdiff_t Extent = gsl::dynamic_extent>
struct PyMatrix {
    using index_type = typename gsl::span<T, Extent>::index_type;

    index_type           rows, cols;
    gsl::span<T, Extent> unpack;
};

template <class T, class IndexType>
auto row2col(IndexType const rows, IndexType const cols,
    gsl::span<T> const row_major, gsl::span<T> const col_major)
{
    for (IndexType i = 0; i < rows; ++i) {
        for (IndexType j = 0; j < cols; ++j) {
            col_major[i + rows * j] = row_major[cols * i + j];
        }
    }
}

template <class T, std::ptrdiff_t Extent, class IndexType>
auto to_matrix(IndexType const rows, IndexType const cols,
    gsl::span<T, Extent> const x) -> PyMatrix<T, Extent>
{
    Expects(rows >= 0 && cols >= 0);
    Expects(rows * cols == x.size());
    return {rows, cols, x};
}

template <class T, std::ptrdiff_t Extent, class IndexType>
auto to_matrix(IndexType const rows, IndexType const cols,
    gsl::span<std::complex<T>, Extent> const x) -> PyMatrix<PyComplex<T>, Extent>
{
    Expects(rows >= 0 && cols >= 0);
    Expects(rows * cols == x.size());
    return {rows, cols,
        gsl::span{reinterpret_cast<PyComplex<T>*>(x.data()), x.size()}};
}

template <class T, std::ptrdiff_t Extent, class IndexType>
auto to_matrix(IndexType const rows, IndexType const cols,
    gsl::span<std::complex<T> const, Extent> const x)
    -> PyMatrix<PyComplex<T> const, Extent>
{
    Expects(rows >= 0 && cols >= 0);
    Expects(rows * cols == x.size());
    return {rows, cols,
        gsl::span{
            reinterpret_cast<PyComplex<T> const*>(x.data()), x.size()}};
}

template <class T, class CharT, class Traits>
auto operator>>(std::basic_istream<CharT, Traits>& is, PyMatrix<T>& x)
    -> std::basic_istream<CharT, Traits>&
{
    Expects(x.rows >= 0);
    Expects(x.cols >= 0);
    Expects(x.rows * x.cols == x.unpack.size());
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        std::ws(is);
        expects(is, CharT{'['});
        if (x.rows > 0) {
            for (auto i = 0; i < x.rows - 1; ++i) {
                auto row = to_list(
                    gsl::span{x.unpack.data() + i * x.cols, x.cols});
                is >> row;
                std::ws(is);
                expects(is, CharT{','});
            }
            auto row = to_list(gsl::span{
                x.unpack.data() + (x.rows - 1) * x.cols, x.cols});
            is >> row;
        }
        std::ws(is);
        expects(is, CharT{']'});
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse PyList<T, Extent>."};
    }
    return is;
}


template <class Rbm>
struct PyRbm {
    Rbm unpack;
};

template <class Rbm, class CharT, class Traits>
auto operator>>(std::basic_istream<CharT, Traits>& is, PyRbm<Rbm>& x)
    -> std::basic_istream<CharT, Traits>&
{
    using vector_type = typename Rbm::vector_type;
    using index_type  = typename Rbm::index_type;
    using C           = typename Rbm::value_type;
    using R           = typename C::value_type;
    auto const io_state = is.exceptions();
    try {
        is.exceptions(std::ios_base::failbit);
        index_type size_visible, size_hidden;
        is >> size_visible >> size_hidden;
        if (size_visible < 0 || size_hidden < 0) {
            throw std::ios_base::failure{
                "RBM can't have negative dimensions."};
        }
        Rbm         rbm{size_visible, size_hidden};
        auto        a = to_list(rbm.visible());
        auto        b = to_list(rbm.hidden());
        vector_type w_data(rbm.size_weights());
        auto        w = to_matrix(
            rbm.size_hidden(), rbm.size_visible(), gsl::span<C>{w_data});
        is >> a >> b >> w;
        row2col(rbm.size_hidden(), rbm.size_visible(),
            gsl::span<C>{w_data}, rbm.weights());
        x.unpack = std::move(rbm);
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse PyList<T, Extent>."};
    }
    return is;
}

template <class Rbm, class CharT, class Traits>
auto parse_rbm_input(std::basic_istream<CharT, Traits>& is) -> Rbm
{
    auto const io_state = is.exceptions();
    Rbm        x{0, 0};
    try {
        is.exceptions(std::ios_base::failbit);
        PyRbm<Rbm> rbm{x};
        is >> rbm;
        x = std::move(rbm.unpack);
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse Rbm."};
    }
    return x;
}

template <class Rbm, class CharT, class Traits>
auto parse_log_wf_input(std::basic_istream<CharT, Traits>& is, Rbm const& rbm)
{
    using vector_type = typename Rbm::vector_type;
    using index_type  = typename Rbm::index_type;
    using C           = typename Rbm::value_type;
    using R           = decltype(std::declval<C>().real());
    auto const                             io_state = is.exceptions();
    std::vector<std::pair<vector_type, C>> expected;
    try {
        is.exceptions(std::ios_base::failbit);
        while (!is.eof()) {
            std::vector<R> spin_real(rbm.size_visible());
            PyList<R>      s{gsl::span<R>{spin_real}};
            PyComplex<R>   psi;
            is >> s >> psi;
            vector_type spin(std::begin(spin_real), std::end(spin_real));
            expected.emplace_back(std::move(spin), psi.unpack);
            std::ws(is);
        }
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse spin - log_wf combinations."};
    }
    return expected;
}

template <class Rbm, class CharT, class Traits>
auto parse_energy_input(std::basic_istream<CharT, Traits>& is)
{
    using index_type = typename Rbm::index_type;
    using C          = typename Rbm::value_type;
    using R          = decltype(std::declval<C>().real());
    auto const                            io_state = is.exceptions();
    std::vector<std::pair<index_type, C>> expected;
    try {
        is.exceptions(std::ios_base::failbit);
        while (!is.eof()) {
            index_type   magnetisation;
            PyComplex<R> energy;
            is >> magnetisation >> energy;
            expected.emplace_back(magnetisation, energy.unpack);
            std::ws(is);
        }
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{
            "Failed to parse magnetisation - energy combinations."};
    }
    return expected;
}

template <class Rbm, class CharT, class Traits>
auto parse_der_log_wf_input(
    std::basic_istream<CharT, Traits>& is, Rbm const& rbm)
{
    using vector_type   = typename Rbm::vector_type;
    using index_type    = typename Rbm::index_type;
    using C             = typename Rbm::value_type;
    using R             = decltype(std::declval<C>().real());
    auto const io_state = is.exceptions();
    std::vector<std::pair<vector_type, vector_type>> expected;
    try {
        is.exceptions(std::ios_base::failbit);
        while (!is.eof()) {
            std::vector<R> spin_real(rbm.size_visible());
            PyList<R>      s{gsl::span<R>{spin_real}};
            vector_type    gradient(rbm.size());
            auto           py_grad = to_list(gsl::make_span(gradient));
            is >> s >> py_grad;
            vector_type spin(std::begin(spin_real), std::end(spin_real));
            expected.emplace_back(std::move(spin), std::move(gradient));
            std::ws(is);
        }
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{"Failed to parse spin - der_log_wf combinations."};
    }
    return expected;
}

template <class Rbm, class CharT, class Traits>
auto parse_force_input(
    std::basic_istream<CharT, Traits>& is, Rbm const& rbm)
{
    using vector_type   = typename Rbm::vector_type;
    using index_type    = typename Rbm::index_type;
    using C             = typename Rbm::value_type;
    using R             = decltype(std::declval<C>().real());
    auto const io_state = is.exceptions();
    std::vector<std::tuple<index_type, C, vector_type>> expected;
    try {
        is.exceptions(std::ios_base::failbit);
        while (!is.eof()) {
            index_type   magnetisation;
            PyComplex<R> py_energy;
            vector_type  force(rbm.size());
            auto         py_force = to_list(gsl::make_span(force));
            is >> magnetisation >> py_energy >> py_force;
            expected.emplace_back(
                magnetisation, py_energy.unpack, std::move(force));
            std::ws(is);
        }
    }
    catch (std::ios_base::failure const& e) {
        is.exceptions(io_state);
        throw parse_error{
            "Failed to parse magnetisation - energy - force combinations."};
    }
    return expected;
}

#endif // TCM_SWARM_PARSE_TEST_HPP
