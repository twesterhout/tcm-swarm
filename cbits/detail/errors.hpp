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

#ifndef TCM_SWARM_DETAIL_ERRORS_HPP
#define TCM_SWARM_DETAIL_ERRORS_HPP

#include <boost/exception/all.hpp>
#include <boost/stacktrace.hpp>

#include "config.hpp"

TCM_SWARM_BEGIN_NAMESPACE

using traced =
    boost::error_info<struct stacktrace_tag, boost::stacktrace::stacktrace>;

template <class E>
[[noreturn]] auto throw_with_trace(E&& e)
{
    throw boost::enable_error_info(std::forward<E>(e))
                           << traced(boost::stacktrace::stacktrace());
}

template <class Int>
class negative_size
    : public virtual boost::exception
    , public virtual std::invalid_argument {

    static_assert(std::is_integral_v<Int>);
    Int x;

  public:
    negative_size(char const* name, Int const value) noexcept(
        std::is_nothrow_default_constructible_v<boost::exception>)
        : boost::exception{}, std::invalid_argument{name}, x{value}
    {
    }
};

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_ERRORS_HPP
