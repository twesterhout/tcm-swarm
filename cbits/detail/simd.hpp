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

#ifndef TCM_SWARM_DETAIL_SIMD_HPP
#define TCM_SWARM_DETAIL_SIMD_HPP

#include <complex>
#include <tuple>

#include <immintrin.h>

#include <Vc/Vc>
#include <gsl/span>

#include "config.hpp"

extern "C" {
#if defined(Vc_HAVE_SSE_ABI)
__m128  __svml_hypotf4(__m128, __m128);
__m128d __svml_hypot2(__m128d, __m128d);
// __m128  vmlsSinCos4(__m128*, __m128);
// __m128d vmldSinCos2(__m128d*, __m128d);
__m128  __svml_sincosf4(__m128);
__m128d __svml_sincos2(__m128d);
__m128  __svml_atan2f4(__m128, __m128);
__m128d __svml_atan22(__m128d, __m128d);
__m128  __svml_fmodf4(__m128, __m128);
__m128d __svml_fmod2(__m128d, __m128d);
__m128  __svml_logf4(__m128);
__m128d __svml_log2(__m128d);
__m128  __svml_clogf2(__m128);
__m128d __svml_clog1(__m128d);
__m128  __svml_log1pf4(__m128);
__m128d __svml_log1p2(__m128d);
__m128  __svml_expf4(__m128);
__m128d __svml_exp2(__m128d);
__m128  __svml_coshf4(__m128);
__m128d __svml_cosh2(__m128d);
__m128  __svml_sinhf4(__m128);
__m128d __svml_sinh2(__m128d);
#endif // Vc_HAVE_FULL_SSE_ABI

#if defined(Vc_HAVE_AVX_ABI)
__m256  __svml_hypotf8(__m256, __m256);
__m256d __svml_hypot4(__m256d, __m256d);
// __m256  vmlsSinCos8(__m256*, __m256);
// __m256d vmldSinCos4(__m256d*, __m256d);
__m256  __svml_sincosf8(__m256);
__m256d __svml_sincos4(__m256d);
__m256  __svml_atan2f8(__m256, __m256);
__m256d __svml_atan24(__m256d, __m256d);
__m256  __svml_fmodf8(__m256, __m256);
__m256d __svml_fmod4(__m256d, __m256d);
__m256  __svml_logf8(__m256);
__m256d __svml_log4(__m256d);
__m256  __svml_clogf4(__m256);
__m256d __svml_clog2(__m256d);
__m256  __svml_log1pf8(__m256);
__m256d __svml_log1p4(__m256d);
__m256  __svml_expf8(__m256);
__m256d __svml_exp4(__m256d);
__m256  __svml_coshf8(__m256);
__m256d __svml_cosh4(__m256d);
__m256  __svml_sinhf8(__m256);
__m256d __svml_sinh4(__m256d);
#endif // Vc_HAVE_FULL_AVX_ABI

#if defined(Vc_HAVE_AVX512_ABI)
// __m512  __svml_hypotf16(__m512, __m512);
// __m512d __svml_hypot8(__m512d, __m512d);
// __m512  vmlsSinCos16(__m512*, __m512); DANGEROUS
// __m512d vmldSinCos8(__m512d*, __m512d); DANGEROUS
// __m512  vmlsAtan216(__m512, __m512);
// __m512d vmldAtan28(__m512d, __m512d);
// __m512  vmlsLog16(__m512);
// __m512d vmldLog8(__m512d);
// __m512  vmlsExp16(__m512);
// __m512d vmldExp8(__m512);
#endif // Vc_HAVE_FULL_AVX512_ABI
} // extern "C"


#if defined(Vc_HAVE_SSE_ABI)
#define MAKE_FN_SSE_FLOAT(macro_name) \
    macro_name(f4,  __m128,  float,  Vc::simd_abi::sse)
#define MAKE_FN_SSE_DOUBLE(macro_name) \
    macro_name(2,   __m128d, double, Vc::simd_abi::sse)
#else // no SSE support
#define MAKE_FN_SSE_FLOAT(macro_name)
#define MAKE_FN_SSE_DOUBLE(macro_name)
#endif // Vc_HAVE_FULL_SSE_ABI

#if defined(Vc_HAVE_AVX_ABI)
#define MAKE_FN_AVX_FLOAT(macro_name) \
    macro_name(f8,  __m256,  float,  Vc::simd_abi::avx)
#define MAKE_FN_AVX_DOUBLE(macro_name) \
    macro_name(4,   __m256d, double, Vc::simd_abi::avx)
#else // no AVX support
#define MAKE_FN_AVX_FLOAT(macro_name)
#define MAKE_FN_AVX_DOUBLE(macro_name)
#endif // Vc_HAVE_FULL_AVX_ABI

#if defined(Vc_HAVE_AVX512_ABI)
#define MAKE_FN_AVX512_FLOAT(macro_name) \
    macro_name(f16, __m512,  float,  Vc::simd_abi::avx512)
#define MAKE_FN_AVX512_DOUBLE(macro_name) \
    macro_name(8,   __m512d, double, Vc::simd_abi::avx512)
#else // no AVX512 support
#define MAKE_FN_AVX512_FLOAT(macro_name)
#define MAKE_FN_AVX512_DOUBLE(macro_name)
#endif // Vc_HAVE_FULL_AVX512_ABI

TCM_SWARM_BEGIN_NAMESPACE

namespace detail {
namespace {
// clang-format off
#define HYPOT_FN(suffix, vector_type, element_type, abi)             \
    auto _hypot(Vc::simd<element_type, abi> const x,                 \
                Vc::simd<element_type, abi> const y) noexcept        \
    {                                                                \
        return static_cast<Vc::simd<element_type, abi>>(             \
            __svml_hypot##suffix(static_cast<vector_type>(x),        \
                static_cast<vector_type>(y)));                       \
    }
    // clang-format on

    MAKE_FN_SSE_FLOAT(HYPOT_FN)
    MAKE_FN_SSE_DOUBLE(HYPOT_FN)
    MAKE_FN_AVX_FLOAT(HYPOT_FN)
    MAKE_FN_AVX_DOUBLE(HYPOT_FN)
    MAKE_FN_AVX512_FLOAT(HYPOT_FN)
    MAKE_FN_AVX512_DOUBLE(HYPOT_FN)

#undef HYPOT_FN
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _hypot(Vc::simd<T, Abi> const x, Vc::simd<T, Abi> const y) noexcept
    -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_hypot(x, y);
}

namespace detail {
namespace {
#define SINH_FN(s, vector_type, element_type, abi)                        \
    auto _sinh(Vc::simd<element_type, abi> const x) noexcept              \
        ->Vc::simd<element_type, abi>                                     \
    {                                                                     \
        return static_cast<Vc::simd<element_type, abi>>(                  \
            __svml_sinh##s(static_cast<vector_type>(x)));                 \
    }

    MAKE_FN_SSE_FLOAT(SINH_FN)
    MAKE_FN_SSE_DOUBLE(SINH_FN)
    MAKE_FN_AVX_FLOAT(SINH_FN)
    MAKE_FN_AVX_DOUBLE(SINH_FN)
    MAKE_FN_AVX512_FLOAT(SINH_FN)
    MAKE_FN_AVX512_DOUBLE(SINH_FN)

#undef SINH_FN
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _sinh(Vc::simd<T, Abi> const x) noexcept -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_sinh(x);
}

namespace detail {
namespace {
#define COSH_FN(s, vector_type, element_type, abi)                        \
    auto _cosh(Vc::simd<element_type, abi> const x) noexcept              \
        ->Vc::simd<element_type, abi>                                     \
    {                                                                     \
        return static_cast<Vc::simd<element_type, abi>>(                  \
            __svml_cosh##s(static_cast<vector_type>(x)));                 \
    }

    MAKE_FN_SSE_FLOAT(COSH_FN)
    MAKE_FN_SSE_DOUBLE(COSH_FN)
    MAKE_FN_AVX_FLOAT(COSH_FN)
    MAKE_FN_AVX_DOUBLE(COSH_FN)
    MAKE_FN_AVX512_FLOAT(COSH_FN)
    MAKE_FN_AVX512_DOUBLE(COSH_FN)

#undef COSH_FN
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _cosh(Vc::simd<T, Abi> const x) noexcept -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_cosh(x);
}

namespace detail {
namespace {
// clang-format off
#define FMOD_FN(s, vector_type, element_type, abi)                        \
    auto _fmod(Vc::simd<element_type, abi> const x,                       \
        Vc::simd<element_type, abi> const y) noexcept                     \
            ->Vc::simd<element_type, abi>                                 \
    {                                                                     \
        return static_cast<Vc::simd<element_type, abi>>(__svml_fmod##s(   \
            static_cast<vector_type>(x), static_cast<vector_type>(y)));   \
    }
// clang-format on

    MAKE_FN_SSE_FLOAT(FMOD_FN)
    MAKE_FN_SSE_DOUBLE(FMOD_FN)
    MAKE_FN_AVX_FLOAT(FMOD_FN)
    MAKE_FN_AVX_DOUBLE(FMOD_FN)
    MAKE_FN_AVX512_FLOAT(FMOD_FN)
    MAKE_FN_AVX512_DOUBLE(FMOD_FN)

#undef FMOD_FN
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _fmod(Vc::simd<T, Abi> const x, Vc::simd<T, Abi> const y) noexcept
    -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_fmod(x, y);
}

namespace detail {
namespace {
#define EXP_FN(s, vector_type, element_type, abi)                    \
    auto _exp(Vc::simd<element_type, abi> const x) noexcept          \
        ->Vc::simd<element_type, abi>                                \
    {                                                                \
        return static_cast<Vc::simd<element_type, abi>>(             \
            __svml_exp##s(static_cast<vector_type>(x)));             \
    }

    MAKE_FN_SSE_FLOAT(EXP_FN)
    MAKE_FN_SSE_DOUBLE(EXP_FN)
    MAKE_FN_AVX_FLOAT(EXP_FN)
    MAKE_FN_AVX_DOUBLE(EXP_FN)
    MAKE_FN_AVX512_FLOAT(EXP_FN)
    MAKE_FN_AVX512_DOUBLE(EXP_FN)

#undef EXP_FN
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _exp(Vc::simd<T, Abi> const x) noexcept -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_exp(x);
}

namespace detail {
namespace {
    TCM_SWARM_FORCEINLINE
    auto _sincos(Vc::simd<float, Vc::simd_abi::sse> const x) noexcept
    {
        __m128 sin_out = __svml_sincosf4(static_cast<__m128>(x));
        __m128 cos_out;
        __asm__ __volatile__("vmovaps %%xmm1, %0" : "=m"(cos_out));
        return std::make_tuple(
            static_cast<Vc::simd<float, Vc::simd_abi::sse>>(sin_out),
            static_cast<Vc::simd<float, Vc::simd_abi::sse>>(cos_out));
    }

    TCM_SWARM_FORCEINLINE
    auto _sincos(Vc::simd<float, Vc::simd_abi::avx> const x) noexcept
    {
        __m256 sin_out = __svml_sincosf8(static_cast<__m256>(x));
        __m256 cos_out;
        __asm__ __volatile__ ("vmovaps %%ymm1, %0":"=m"(cos_out));
        return std::make_tuple(
            static_cast<Vc::simd<float, Vc::simd_abi::avx>>(sin_out),
            static_cast<Vc::simd<float, Vc::simd_abi::avx>>(cos_out));
    }

    TCM_SWARM_FORCEINLINE
    auto _sincos(Vc::simd<double, Vc::simd_abi::sse> const x) noexcept
    {
        __m128d sin_out = __svml_sincos2(static_cast<__m128d>(x));
        __m128d cos_out;
        __asm__ __volatile__("vmovaps %%xmm1, %0" : "=m"(cos_out));
        return std::make_tuple(
            static_cast<Vc::simd<double, Vc::simd_abi::sse>>(sin_out),
            static_cast<Vc::simd<double, Vc::simd_abi::sse>>(cos_out));
    }

    TCM_SWARM_FORCEINLINE
    auto _sincos(Vc::simd<double, Vc::simd_abi::avx> const x) noexcept
    {
        __m256d sin_out = __svml_sincos4(static_cast<__m256d>(x));
        __m256d cos_out;
        __asm__ __volatile__("vmovaps %%ymm1, %0" : "=m"(cos_out));
        return std::make_tuple(
            static_cast<Vc::simd<double, Vc::simd_abi::avx>>(sin_out),
            static_cast<Vc::simd<double, Vc::simd_abi::avx>>(cos_out));
    }
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
auto _sincos(Vc::simd<T, Abi> const x) noexcept
    -> std::tuple<Vc::simd<T, Abi>, Vc::simd<T, Abi>>
// clang-format on
{
    return detail::_sincos(x);
}

namespace detail {
namespace {
// clang-format off
#define ATAN2_FN(s, vector_type, element_type, abi)                  \
    auto _atan2(Vc::simd<element_type, abi> const x,                 \
        Vc::simd<element_type, abi> const y) noexcept                \
            ->Vc::simd<element_type, abi>                            \
    {                                                                \
        return static_cast<Vc::simd<element_type, abi>>(             \
            __svml_atan2##s(static_cast<vector_type>(x),             \
                static_cast<vector_type>(y)));                       \
    }
    // clang-format on

    MAKE_FN_SSE_FLOAT(ATAN2_FN)
    MAKE_FN_SSE_DOUBLE(ATAN2_FN)
    MAKE_FN_AVX_FLOAT(ATAN2_FN)
    MAKE_FN_AVX_DOUBLE(ATAN2_FN)
    MAKE_FN_AVX512_FLOAT(ATAN2_FN)
    MAKE_FN_AVX512_DOUBLE(ATAN2_FN)

#undef ATAN2_FN
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _atan2(Vc::simd<T, Abi> const x, Vc::simd<T, Abi> const y)
    noexcept -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_atan2(x, y);
}

namespace detail {
namespace {
    // MAKE_FN_SSE_FLOAT(CLOG_FN)
    // MAKE_FN_SSE_DOUBLE(CLOG_FN)

    auto _clog(Vc::simd<float, Vc::simd_abi::sse> const x) noexcept
        -> Vc::simd<float, Vc::simd_abi::sse>
    {
        return static_cast<Vc::simd<float, Vc::simd_abi::sse>>(
            __svml_clogf2(static_cast<__m128>(x)));
    }

    auto _clog(Vc::simd<float, Vc::simd_abi::avx> const x) noexcept
        -> Vc::simd<float, Vc::simd_abi::avx>
    {
        return static_cast<Vc::simd<float, Vc::simd_abi::avx>>(
            __svml_clogf4(static_cast<__m256>(x)));
    }

    // MAKE_FN_AVX_DOUBLE(CLOG_FN)
    // MAKE_FN_AVX512_FLOAT(CLOG_FN)
    // MAKE_FN_AVX512_DOUBLE(CLOG_FN)

} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _clog(Vc::simd<T, Abi> const x) noexcept -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_clog(x);
}

namespace detail {
namespace {
#define LOG_FN(s, vector_type, element_type, abi)                    \
    auto _log(Vc::simd<element_type, abi> const x) noexcept          \
        ->Vc::simd<element_type, abi>                                \
    {                                                                \
        return static_cast<Vc::simd<element_type, abi>>(             \
            __svml_log##s(static_cast<vector_type>(x)));             \
    }

    MAKE_FN_SSE_FLOAT(LOG_FN)
    MAKE_FN_SSE_DOUBLE(LOG_FN)
    MAKE_FN_AVX_FLOAT(LOG_FN)
    MAKE_FN_AVX_DOUBLE(LOG_FN)
    MAKE_FN_AVX512_FLOAT(LOG_FN)
    MAKE_FN_AVX512_DOUBLE(LOG_FN)

#undef LOG_FN
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _log(Vc::simd<T, Abi> const x) noexcept -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_log(x);
}

namespace detail {
namespace {
#define LOG1P_FN(s, vector_type, element_type, abi)                       \
    auto _log1p(Vc::simd<element_type, abi> const x) noexcept             \
        ->Vc::simd<element_type, abi>                                     \
    {                                                                     \
        return static_cast<Vc::simd<element_type, abi>>(                  \
            __svml_log1p##s(static_cast<vector_type>(x)));                \
    }

    MAKE_FN_SSE_FLOAT(LOG1P_FN)
    MAKE_FN_SSE_DOUBLE(LOG1P_FN)
    MAKE_FN_AVX_FLOAT(LOG1P_FN)
    MAKE_FN_AVX_DOUBLE(LOG1P_FN)
    MAKE_FN_AVX512_FLOAT(LOG1P_FN)
    MAKE_FN_AVX512_DOUBLE(LOG1P_FN)

#undef LOG1P_FN
} // namespace
} // namespace detail

// clang-format off
template <class T, class Abi>
TCM_SWARM_ARTIFIFICAL
TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _log1p(Vc::simd<T, Abi> const x) noexcept -> Vc::simd<T, Abi>
// clang-format on
{
    return detail::_log1p(x);
}

template <class T, class Abi>
TCM_SWARM_FORCEINLINE auto _to_min_pi_pi(
    Vc::simd<T, Abi> const x) noexcept
{
#if 1
#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#endif
    constexpr T pi{
        3.141592653589793238462643383279502884197169399375105821};
    constexpr T two_pi{
        6.283185307179586476925286766559005768394338798750211642};
#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic push
#endif
    auto y = _fmod(x, Vc::simd<T, Abi>{two_pi});
    Vc::where(y >  pi, y) -= two_pi;
    Vc::where(y < -pi, y) += two_pi;
    return y;
#else
    using V = Vc::simd<float, Vc::simd_abi::avx>;

    constexpr float one_over_pi{0.318309886183790671537767526745028724};
    constexpr float pi_part_1{3.140625f};
    constexpr float pi_part_2{0.0009670257568359375f};
    constexpr float pi_part_3{6.2771141529083251953e-07f};
    constexpr float pi_part_4{1.2154201256553420762e-10f};

    auto q = static_cast<V>(_mm256_round_ps(
        static_cast<__m256>(x * one_over_pi), _MM_FROUND_TO_ZERO));
    // add one to all odd numbers
    // no idea how to do it though...
    q += static_cast<V>(_mm256_and_ps(
        static_cast<__m256>(q), _mm256_set1_ps(1.0f)));
    std::cout << "q'= " << q << '\n';
    auto y = x;
    y -= q * pi_part_1;
    y -= q * pi_part_2;
    y -= q * pi_part_3;
    y -= q * pi_part_4;
    return y;
#endif
}

namespace detail {
template <class Flags>
TCM_SWARM_FORCEINLINE auto copy_from(std::complex<float> const* p, Flags flags,
    Vc::simd_abi::sse /*unused*/) noexcept
    -> std::complex<Vc::simd<float, Vc::simd_abi::sse>>
{
    static_assert(std::is_empty_v<Vc::simd_abi::sse>, "");
    using V          = Vc::simd<float, Vc::simd_abi::sse>;
    auto const* data = reinterpret_cast<float const*>(p);
    V const     x{data, flags};
    V const     y{data + V::size(), flags};
    auto const  real = static_cast<V>(_mm_shuffle_ps(static_cast<__m128>(x),
        static_cast<__m128>(y), _MM_SHUFFLE(2, 0, 2, 0)));
    auto const  imag = static_cast<V>(_mm_shuffle_ps(static_cast<__m128>(x),
        static_cast<__m128>(y), _MM_SHUFFLE(3, 1, 3, 1)));
    return {real, imag};
}

template <class Flags>
TCM_SWARM_FORCEINLINE auto copy_from(std::complex<float> const* p, Flags flags,
    Vc::simd_abi::avx /*unused*/) noexcept
    -> std::complex<Vc::simd<float, Vc::simd_abi::avx>>
{
    static_assert(std::is_empty_v<Vc::simd_abi::avx>, "");
    using V          = Vc::simd<float, Vc::simd_abi::avx>;
    auto const* data = reinterpret_cast<float const*>(p);
    V const     x{data, flags};
    V const     y{data + V::size(), flags};
    auto const  real = static_cast<V>(_mm256_shuffle_ps(static_cast<__m256>(x),
        static_cast<__m256>(y), _MM_SHUFFLE(2, 0, 2, 0)));
    auto const  imag = static_cast<V>(_mm256_shuffle_ps(static_cast<__m256>(x),
        static_cast<__m256>(y), _MM_SHUFFLE(3, 1, 3, 1)));
    return {real, imag};
}

template <class Flags>
TCM_SWARM_FORCEINLINE auto copy_to(
    std::complex<Vc::simd<float, Vc::simd_abi::sse>> const z,
    std::complex<float>* p, Flags flags) noexcept -> void
{
    using V         = Vc::simd<float, Vc::simd_abi::sse>;
    auto*      data = reinterpret_cast<float*>(p);
    auto const a    = static_cast<V>(_mm_unpacklo_ps(
        static_cast<__m128>(z.real()), static_cast<__m128>(z.imag())));
    auto const b    = static_cast<V>(_mm_unpackhi_ps(
        static_cast<__m128>(z.real()), static_cast<__m128>(z.imag())));
    a.copy_to(data, flags);
    b.copy_to(data + V::size(), flags);
}

template <class Flags>
TCM_SWARM_FORCEINLINE auto copy_to(
    std::complex<Vc::simd<float, Vc::simd_abi::avx>> const z,
    std::complex<float>* p, Flags flags) noexcept -> void
{
    using V         = Vc::simd<float, Vc::simd_abi::avx>;
    auto*      data = reinterpret_cast<float*>(p);
    auto const a    = static_cast<V>(_mm256_unpacklo_ps(
        static_cast<__m256>(z.real()), static_cast<__m256>(z.imag())));
    auto const b    = static_cast<V>(_mm256_unpackhi_ps(
        static_cast<__m256>(z.real()), static_cast<__m256>(z.imag())));
    a.copy_to(data, flags);
    b.copy_to(data + V::size(), flags);
}
} // namespace detail

template <class Abi, class T, class Flags>
TCM_SWARM_FORCEINLINE auto copy_from(std::complex<T> const* p,
    Flags flags) noexcept -> std::complex<Vc::simd<T, Abi>>
{
    return detail::copy_from(p, flags, Abi{});
}

template <class Abi, class T, class Flags>
TCM_SWARM_FORCEINLINE auto copy_to(std::complex<Vc::simd<T, Abi>> const z,
    std::complex<T>* p, Flags flags) noexcept -> void
{
    detail::copy_to(z, p, flags);
}

TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _interleave(Vc::simd<float, Vc::simd_abi::sse> const x,
    Vc::simd<float, Vc::simd_abi::sse> const              y) noexcept
    -> std::tuple<Vc::simd<float, Vc::simd_abi::sse>,
        Vc::simd<float, Vc::simd_abi::sse>>
{
    auto const a = static_cast<Vc::simd<float, Vc::simd_abi::sse>>(
        _mm_unpacklo_ps(static_cast<__m128>(x), static_cast<__m128>(y)));
    auto const b = static_cast<Vc::simd<float, Vc::simd_abi::sse>>(
        _mm_unpackhi_ps(static_cast<__m128>(x), static_cast<__m128>(y)));
    return {a, b};
}

TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _interleave(Vc::simd<float, Vc::simd_abi::avx> const x,
    Vc::simd<float, Vc::simd_abi::avx> const              y) noexcept
    -> std::tuple<Vc::simd<float, Vc::simd_abi::avx>,
        Vc::simd<float, Vc::simd_abi::avx>>
{
    auto const a =
        static_cast<Vc::simd<float, Vc::simd_abi::avx>>(_mm256_unpacklo_ps(
            static_cast<__m256>(x), static_cast<__m256>(y)));
    auto const b =
        static_cast<Vc::simd<float, Vc::simd_abi::avx>>(_mm256_unpackhi_ps(
            static_cast<__m256>(x), static_cast<__m256>(y)));
    return {a, b};
}

TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _deinterleave(Vc::simd<float, Vc::simd_abi::sse> const x,
    Vc::simd<float, Vc::simd_abi::sse> const              y) noexcept
    -> std::tuple<Vc::simd<float, Vc::simd_abi::sse>,
        Vc::simd<float, Vc::simd_abi::sse>>
{
    auto const a = static_cast<Vc::simd<float, Vc::simd_abi::sse>>(
        _mm_shuffle_ps(static_cast<__m128>(x), static_cast<__m128>(y),
            _MM_SHUFFLE(2, 0, 2, 0)));
    auto const b = static_cast<Vc::simd<float, Vc::simd_abi::sse>>(
        _mm_shuffle_ps(static_cast<__m128>(x), static_cast<__m128>(y),
            _MM_SHUFFLE(3, 1, 3, 1)));
    return {a, b};
}

TCM_SWARM_FORCEINLINE
TCM_SWARM_CONST
auto _deinterleave(Vc::simd<float, Vc::simd_abi::avx> const x,
    Vc::simd<float, Vc::simd_abi::avx> const              y) noexcept
    -> std::tuple<Vc::simd<float, Vc::simd_abi::avx>,
        Vc::simd<float, Vc::simd_abi::avx>>
{
    auto const a = static_cast<Vc::simd<float, Vc::simd_abi::avx>>(
        _mm256_shuffle_ps(static_cast<__m256>(x), static_cast<__m256>(y),
            _MM_SHUFFLE(2, 0, 2, 0)));
    auto const b = static_cast<Vc::simd<float, Vc::simd_abi::avx>>(
        _mm256_shuffle_ps(static_cast<__m256>(x), static_cast<__m256>(y),
            _MM_SHUFFLE(3, 1, 3, 1)));
    return {a, b};
}

template <class T>
TCM_SWARM_FORCEINLINE auto _clog(Vc::simd<T, Vc::simd_abi::sse> const x,
    Vc::simd<T, Vc::simd_abi::sse> const y) noexcept
{
    auto [a, b] = _interleave(x, y);
    return _deinterleave(_clog(a), _clog(b));
}

template <class T>
TCM_SWARM_FORCEINLINE auto _clog(Vc::simd<T, Vc::simd_abi::avx> const x,
    Vc::simd<T, Vc::simd_abi::avx> const y) noexcept
    -> std::tuple<Vc::simd<T, Vc::simd_abi::avx>,
        Vc::simd<T, Vc::simd_abi::avx>>
{
    auto [a, b] = _interleave(x, y);
    return _deinterleave(_clog(a), _clog(b));
    // return {T{0.5} * _log(x * x + y * y), _atan2(y, x)};
}


template <class T, class Abi>
TCM_SWARM_FORCEINLINE auto _log_cosh(
    Vc::simd<T, Abi> x, Vc::simd<T, Abi> y) noexcept
    -> std::tuple<Vc::simd<T, Abi>, Vc::simd<T, Abi>>
{
    constexpr auto cutoff = T{20.0};
    static auto const log_of_2 = Vc::simd<T, Abi>{
        T{0.69314718055994530941723212145817656807550013436025525412068}};

    auto const smaller_than_zero = x < T{0};
    Vc::where(smaller_than_zero, x) *= T{-1};
    Vc::where(smaller_than_zero, y) *= T{-1};

    if (Vc::all_of(x > cutoff)) {
        return {x - log_of_2, _to_min_pi_pi(y)};
    }
    else {
        auto const exp_min_2x     = _exp(-T{2} * x);
        auto const [sin_y, cos_y] = _sincos(y);
        auto const [real, imag]   = _clog(
            cos_y + cos_y * exp_min_2x, sin_y * (T{1} - exp_min_2x));
        return {x - (log_of_2 - real), imag};
    }
}

template <class T, class Abi>
auto _log_cosh(std::complex<Vc::simd<T, Abi>> const z) noexcept(
    !detail::gsl_can_throw()) -> std::complex<Vc::simd<T, Abi>>
{
    auto const [x, y] = _log_cosh(z.real(), z.imag());
    return {x, y};
}

template <class T, class Abi = Vc::simd_abi::native<T>,
    class = std::enable_if_t<std::is_floating_point_v<T>>>
auto _log_cosh(std::complex<T> const z) noexcept(!detail::gsl_can_throw())
    -> std::complex<T>
{
    using V = Vc::simd<T, Abi>;
    constexpr auto vector_size = V::size();
    V x{0}, y{0};
    x[0] = z.real();
    y[0] = z.imag();
    // auto const [a, b] = _log_cosh(x, y);
    std::tie(x, y) = _log_cosh(x, y);
    return {x[0], y[0]};
}

/*
template <class T, class Abi>
TCM_SWARM_FORCEINLINE auto _logcosh(
    Vc::simd<T, Abi> x, Vc::simd<T, Abi> y) noexcept
    -> std::tuple<Vc::simd<T, Abi>, Vc::simd<T, Abi>>
{
    auto const [sin_y, cos_y] = _sincos(y);
    return _clog(cos_y * _cosh(x), sin_y * _sinh(x));
}
*/

template <class T, class Abi = Vc::simd_abi::native<T>>
auto TCM_SWARM_NOINLINE sum_log_cosh(
    gsl::span<std::complex<T> const> const x) noexcept(!detail::gsl_can_throw())
{
#if 0
    std::complex<T> sum{0};
    for (auto y : x) {
        if (std::abs(y.real()) > 20.0) {
            std::cout << "[";
            std::copy(std::begin(x), std::end(x),
                std::ostream_iterator<std::complex<T>>{std::cout, ", "});
            std::cout << "]\n";
            std::cout << y << '\n';
            Expects(false);
        }
        sum += std::log(std::cosh(y));
    }
    return sum;
#else
    using V = Vc::simd<T, Abi>;
    using index_type =
        typename gsl::span<std::complex<T> const>::index_type;
    auto constexpr vector_size = static_cast<index_type>(V::size());
    auto            count      = x.size() / vector_size;
    auto            rest       = x.size() % vector_size;
    V               sum_real   = 0;
    V               sum_imag   = 0;
    std::complex<T> sum_rest   = 0;
    auto const*     data       = x.data();
    for (; count != 0; --count, data += vector_size) {
        auto [a, b] = tcm::_deinterleave(
            V{reinterpret_cast<T const*>(data), Vc::flags::element_aligned},
            V{reinterpret_cast<T const*>(data + vector_size / 2),
                Vc::flags::element_aligned});
        std::tie(a, b) = tcm::_log_cosh(a, b);
        sum_real += a;
        sum_imag += b;
    }
    for (; rest != 0; --rest, ++data) {
        sum_rest += _log_cosh(*data);
    }
    return std::complex{Vc::reduce(sum_real), Vc::reduce(sum_imag)}
           + sum_rest;
#endif
}

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_SIMD_HPP
