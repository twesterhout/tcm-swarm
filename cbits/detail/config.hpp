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

#ifndef TCM_SWARM_DETAIL_CONFIG_HPP
#define TCM_SWARM_DETAIL_CONFIG_HPP

#include <cstddef>
#include <gsl/gsl_assert>

#define TCM_SWARM_NAMESPACE tcm

#define TCM_SWARM_BEGIN_NAMESPACE namespace tcm {

#define TCM_SWARM_END_NAMESPACE } /* tcm */

#if defined(__clang__)
// ===========================================================================
// We're being compiled with Clang
#define TCM_SWARM_CLANG                                              \
    (__clang_major__ * 10000 + __clang_minor__ * 100                 \
        + __clang_patchlevel__)

#define TCM_SWARM_FORCEINLINE                                        \
    inline __attribute__((__always_inline__))

#define TCM_SWARM_NOINLINE __attribute__((__noinline__))

#define TCM_SWARM_UNUSED __attribute__((__unused__))

#define TCM_SWARM_NORETURN __attribute__((__noreturn__))

#define TCM_SWARM_CONST __attribute__((__const__))

#define TCM_SWARM_PURE __attribute__((__pure__))

#define TCM_SWARM_ARTIFIFICAL __attribute__((__artificial__))

#define TCM_SWARM_LIKELY(cond) __builtin_expect(!!(cond), 1)

#define TCM_SWARM_UNLIKELY(cond) __builtin_expect(!!(cond), 0)

#define TCM_SWARM_ASSUME(cond) __builtin_assume(!!(cond))

#define TCM_SWARM_CURRENT_FUNCTION __PRETTY_FUNCTION__

#define TCM_SWARM_UNREACHABLE __builtin_unreachable()

#elif defined(__GNUC__)
// ===========================================================================
// We're being compiled with GCC
#define TCM_SWARM_GCC                                                \
    (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#define TCM_SWARM_FORCEINLINE                                        \
    inline __attribute__((__always_inline__))

#define TCM_SWARM_NOINLINE __attribute__((__noinline__))

#define TCM_SWARM_UNUSED __attribute__((__unused__))

#define TCM_SWARM_NORETURN __attribute__((__noreturn__))

#define TCM_SWARM_CONST __attribute__((__const__))

#define TCM_SWARM_PURE __attribute__((__pure__))

#define TCM_SWARM_ARTIFIFICAL __attribute__((__artificial__))

#define TCM_SWARM_LIKELY(cond) __builtin_expect(!!(cond), 1)

#define TCM_SWARM_UNLIKELY(cond) __builtin_expect(!!(cond), 0)

#define TCM_SWARM_ASSUME(cond)                                       \
    ((!!(cond)) ? static_cast<void>(0) : __builtin_unreachable())

#define TCM_SWARM_CURRENT_FUNCTION __PRETTY_FUNCTION__

#define TCM_SWARM_UNREACHABLE __builtin_unreachable()

#elif defined(_MSC_VER)
// ===========================================================================
// We're being compiled with Microsoft Visual C++
#define TCM_SWARM_MSVC _MSC_VER

#define TCM_SWARM_FORCEINLINE inline __forceinline

#define TCM_SWARM_NOINLINE

#define TCM_SWARM_UNUSED

#define TCM_SWARM_NORETURN __declspec(noreturn)

#define TCM_SWARM_CONST

#define TCM_SWARM_PURE

#define TCM_SWARM_ARTIFIFICAL

#define TCM_SWARM_LIKELY(cond) (cond)

#define TCM_SWARM_UNLIKELY(cond) (cond)

#define TCM_SWARM_ASSUME(cond) __assume(!!(cond))

#define TCM_SWARM_CURRENT_FUNCTION __FUNCTION__

#define TCM_SWARM_UNREACHABLE __assume(0)

#else
// clang-format off
#   error "Unsupported compiler. Please, submit a request to https://github.com/twesterhout/tcm-swarm/issues."
// clang-format on
// ===========================================================================
#endif

TCM_SWARM_BEGIN_NAMESPACE

/// \cond IMPLEMENTATION
// See
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4381.html
template <class T>
constexpr T _static_const{};
/// \endcond

TCM_SWARM_END_NAMESPACE

#if defined(DOXYGEN_IN_HOUSE)
    constexpr type name{};
#else
#define TCM_SWARM_INLINE_VARIABLE(type, name)                        \
    inline namespace {                                               \
        constexpr auto const& name =                                 \
            ::TCM_SWARM_NAMESPACE::_static_const<type>;              \
    }                                                                \
    /**/
#endif

TCM_SWARM_BEGIN_NAMESPACE

namespace detail {
    constexpr auto gsl_can_throw() noexcept -> bool
    {
#if defined(GSL_THROW_ON_CONTRACT_VIOLATION)
        return true;
#elif defined(GSL_TERMINATE_ON_CONTRACT_VIOLATION)                        \
    || defined(GSL_TERMINATE_ON_CONTRACT_VIOLATION)
        return false;
#else
#error "BUG! This function assumes that <gsl/gsl_assert> is included!"
#endif
    }
} // namespace detail

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_CONFIG_HPP
