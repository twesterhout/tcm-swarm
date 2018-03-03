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

#ifndef TCM_SWARM_DETAIL_RANDOM_HPP
#define TCM_SWARM_DETAIL_RANDOM_HPP

#include <cstdio>
#include <random>

#include "../detail/config.hpp"
#include "../detail/mkl.hpp"

#include <gsl/pointers>
#include <gsl/span>

TCM_SWARM_BEGIN_NAMESPACE

TCM_SWARM_NOINLINE
auto really_need_that_random_seed_now() -> std::uint64_t
{
    static thread_local std::random_device random_device;
    static thread_local std::uniform_int_distribution<std::uint64_t> dist;
    return dist(random_device);
}

namespace mkl {

enum class GenType : MKL_INT {
    mcg31         = VSL_BRNG_MCG31,
    r250          = VSL_BRNG_R250,
    mrg32k3a      = VSL_BRNG_MRG32K3A,
    mcg59         = VSL_BRNG_MCG59,
    wh            = VSL_BRNG_WH,
    sobol         = VSL_BRNG_SOBOL,
    niederr       = VSL_BRNG_NIEDERR,
    mt19937       = VSL_BRNG_MT19937,
    mt2203        = VSL_BRNG_MT2203,
    // iabstract     = VSL_BRNG_IABSTRACT,
    // dabstract     = VSL_BRNG_DABSTRACT,
    // sabstract     = VSL_BRNG_SABSTRACT,
    sfmt19937     = VSL_BRNG_SFMT19937,
    // nondeterm     = VSL_BRNG_NONDETERM,
    // ars5          = VSL_BRNG_ARS5,
    philox4x32x10 = VSL_BRNG_PHILOX4X32X10
};

template <class T>
constexpr auto swap(T*& x, T*& y) noexcept -> void
{
    auto const t = x;
    x            = y;
    y            = t;
}

struct random_generator {
    using state_ptr_t = VSLStreamStatePtr;
    static_assert(std::is_pointer_v<state_ptr_t>,
        "VSLStreamStatePtr is not a pointer.");

  private:
    [[nodiscard]] static auto _make_generator(GenType const generator,
        size_type const seed) -> gsl::owner<state_ptr_t>
    {
        state_ptr_t p;
        auto const  err = vslNewStream(&p,
            static_cast<std::underlying_type_t<GenType>>(generator), seed);
        if (TCM_SWARM_UNLIKELY(err != VSL_ERROR_OK)) {
            switch (err) {
            case VSL_ERROR_MEM_FAILURE: throw std::bad_alloc{};
            case VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED:
                throw std::runtime_error{
                    "tcm::mkl::random_generator::_make_generator(): "
                    "Non-deterministic generator is not supported."};
            case VSL_RNG_ERROR_ARS5_NOT_SUPPORTED:
                throw std::runtime_error{
                    "tcm::mkl::random_generator::_make_generator(): "
                    "ARS5 generator is not supported."};
            }
        }
        Ensures(p != nullptr);
        return p;
    }

  public:
    explicit random_generator(GenType const generator, size_type const seed)
        : _handle{_make_generator(generator, seed)}
    {
        Ensures(_handle != nullptr);
    }

    random_generator(random_generator const&) = delete;

    constexpr random_generator(random_generator&& other) noexcept
        : _handle{other._handle}
    {
        Expects(other._handle != nullptr);
        other._handle = nullptr;
        Ensures(_handle != nullptr);
    }

    random_generator& operator=(random_generator const&) = delete;
    constexpr random_generator& operator=(random_generator&&) noexcept = delete;

    constexpr auto get() -> gsl::not_null<state_ptr_t>
    {
        return _handle;
    }

    ~random_generator()
    {
        if (_handle != nullptr) {
            vslDeleteStream(&_handle);
        }
    }

  private:
    state_ptr_t _handle;
};

enum class Method : MKL_INT {
    uniform_standard = VSL_RNG_METHOD_UNIFORM_STD,
    uniform_accurate = VSL_RNG_METHOD_UNIFORM_STD_ACCURATE,
};

namespace detail {
    // clang-format off
    [[nodiscard]] TCM_SWARM_FORCEINLINE
    auto uniform(std::underlying_type_t<Method> const method,
        random_generator::state_ptr_t const stream,
        difference_type const n, float* const r, float const a,
        float const b)
    // clang-format on
    {
        return vsRngUniform(method, stream, n, r, a, b);
    }

    // clang-format off
    template <GenType generator>
    [[nodiscard]] TCM_SWARM_FORCEINLINE
    auto uniform(std::underlying_type_t<Method> const method,
        random_generator::state_ptr_t const stream,
        difference_type const n, double* const r, double const a,
        double const b)
    // clang-format on
    {
        return vdRngUniform(method, stream, n, r, a, b);
    }

    // clang-format off
    [[nodiscard]] TCM_SWARM_FORCEINLINE
    auto uniform(std::underlying_type_t<Method> const method,
        random_generator::state_ptr_t const stream,
        difference_type const n, int* const r, int const a,
        int const b)
    // clang-format on
    {
        return viRngUniform(method, stream, n, r, a, b);
    }

    // clang-format off
    [[nodiscard]] TCM_SWARM_FORCEINLINE
    auto uniform(std::underlying_type_t<Method> const method,
        random_generator::state_ptr_t const stream,
        difference_type const n, std::complex<float>* const r,
        float const a, float const b)
    // clang-format on
    {
        static_assert(sizeof(std::complex<float>) == 2 * sizeof(float));
        return vsRngUniform(
            method, stream, 2 * n, reinterpret_cast<float*>(r), a, b);
    }
} // namespace detail

struct uniform_fn {
    template <class T, class U>
    auto operator()(Method const method, random_generator& stream,
        gsl::span<T> r, U const a, U const b) const
    {
        auto const err = detail::uniform(
            static_cast<std::underlying_type_t<Method>>(method),
            stream.get(), static_cast<difference_type>(r.size()), r.data(),
            a, b);
        if (TCM_SWARM_UNLIKELY(err != VSL_ERROR_OK)) {
            if (err == VSL_RNG_ERROR_QRNG_PERIOD_ELAPSED) {
                throw std::runtime_error{
                    "tcm::mkl::uniform_fn::operator(): "
                    "Period of the generator has been exceeded."};
            }
            else {
                throw std::runtime_error{
                    "tcm::mkl::uniform_fn::operator(): Bug! "
                    "Unreachable code reached!"};
            }
        }
    }
};

TCM_SWARM_INLINE_VARIABLE(uniform_fn, uniform)

template <class T>
struct random_stream {
  private:
    auto refill()
    {
        mkl::uniform(mkl::Method::uniform_standard, _generator, _buffer,
            _min, _max);
    }

  public:
    random_stream(mkl::random_generator& generator, T min, T max,
        gsl::span<T> buffer) noexcept
        : _generator{generator}, _buffer{buffer}, _min{min}, _max{max}, _i{0}
    {
        refill();
    }

    friend
    auto operator>>(random_stream& stream, T& x) -> random_stream&
    {
        Expects(stream._i < stream._buffer.size());
        x = stream._buffer[stream._i];
        if (++stream._i == stream._buffer.size()) {
            stream._i = 0;
            stream.refill();
        }
    }

  private:
    mkl::random_generator& _generator;
    gsl::span<T>           _buffer;
    T                      _min;
    T                      _max;
    mkl::difference_type   _i;
};



} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_RANDOM_HPP

